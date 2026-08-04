"""Rolling-origin forecasts for all models.
Origins: monthly, 2022-01..2024-12; at origin t data through t-1 is known.
Horizon: months t..t+4 (j=0..4), i.e. steps 1..5 ahead. Forecasts clipped >=0.
Local models refit every origin; global ML models retrained yearly (vintages).
"""
import pandas as pd, numpy as np, json, os, warnings
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
panel = pd.read_csv(os.path.join(HERE, 'unit_demand_monthly.csv'), index_col=0, parse_dates=True)
mitra = json.load(open(os.path.join(HERE, 'mitra_annual.json')))
UNITS = list(panel.columns)
MONTHS = list(panel.index)
ORIGINS = [m for m in MONTHS if m >= pd.Timestamp('2022-01-01')]

# ---------- exogenous ----------
exog = pd.read_csv(os.path.join(HERE, 'exog_monthly.csv'), index_col=0, parse_dates=True)

# ---------- local methods ----------
def ses(h, y):
    best = (None, np.inf)
    for a in np.arange(0.05, 1.0, 0.05):
        f, s = [], y[0]
        for v in y[1:]:
            f.append(s); s = a * v + (1 - a) * s
        e = np.mean((np.array(f) - np.array(y[1:]))**2) if f else np.inf
        if e < best[1]: best = (s, e)
    return max(best[0], 0.0)

def croston_variants(y, alpha=0.1):
    """returns dict method-> per-period forecast rate (flat)."""
    y = np.asarray(y, float)
    nz = np.nonzero(y)[0]
    if len(nz) == 0:
        return {'croston': 0.0, 'sba': 0.0, 'tsb': 0.0}
    z = y[nz[0]]; p = 1.0; first = True
    q = 1
    for i in range(nz[0] + 1, len(y)):
        if y[i] > 0:
            z = alpha * y[i] + (1 - alpha) * z
            p = alpha * q + (1 - alpha) * p if not first else q
            first = False
            q = 1
        else:
            q += 1
    cro = z / max(p, 1.0)
    sba = cro * (1 - alpha / 2)
    # TSB
    prob = 1.0 if y[0] > 0 else 0.0
    zt = y[nz[0]]
    for i in range(1, len(y)):
        if y[i] > 0:
            prob = 0.1 * 1 + 0.9 * prob
            zt = 0.1 * y[i] + 0.9 * zt
        else:
            prob = 0.9 * prob
    return {'croston': cro, 'sba': sba, 'tsb': prob * zt}

from statsmodels.tsa.arima.model import ARIMA as SM_ARIMA
def arima_fc(y, steps=5):
    best_aic, best_fc = np.inf, None
    for p in range(3):
        for d in range(2):
            for q in range(3):
                if p == 0 and q == 0 and d == 0: continue
                try:
                    m = SM_ARIMA(y, order=(p, d, q)).fit(method_kwargs={'maxiter': 50})
                    if m.aic < best_aic:
                        best_aic = m.aic
                        best_fc = np.clip(m.forecast(steps), 0, None)
                except Exception:
                    continue
    if best_fc is None:
        best_fc = np.repeat(np.mean(y[-12:]), steps)
    return np.asarray(best_fc, float)

# ---------- global ML (vintages) ----------
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier

def make_features(minlag):
    """long df: one row per (unit, month) with lags minlag..minlag+3 etc."""
    frames = []
    for u in UNITS:
        s = panel[u]
        d = pd.DataFrame({'y': s})
        for k in range(minlag, minlag + 4):
            d[f'lag{k}'] = s.shift(k)
        d['ma3'] = s.shift(minlag).rolling(3).mean()
        d['month'] = d.index.month
        d['unit'] = u
        d['type_broken'] = 1 if u.startswith('BR') else 0
        ex = exog.reindex(d.index).shift(minlag)
        d = pd.concat([d, ex.add_prefix('ex_')], axis=1)
        frames.append(d.reset_index().rename(columns={'index': 'Date'}))
    long = pd.concat(frames, ignore_index=True)
    long = pd.get_dummies(long, columns=['month', 'unit'], dtype=int)
    return long.dropna(subset=[f'lag{minlag}', f'lag{minlag+3}', 'ma3'])

FEATCOLS = {}
LONGS = {}
for minlag in (4, 5):
    L = make_features(minlag)
    LONGS[minlag] = L
    FEATCOLS[minlag] = [c for c in L.columns if c not in ('y', 'Date')]

def train_vintage(minlag, cutoff):
    L = LONGS[minlag]
    tr = L[L.Date <= cutoff]
    X, y = tr[FEATCOLS[minlag]].astype(float), tr['y'].astype(float)
    models = {}
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1).fit(X, y)
    xgb = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9,
                       colsample_bytree=0.9, random_state=42, n_jobs=-1).fit(X, y)
    ybin = (y > 0).astype(int)
    clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42,
                        n_jobs=-1, eval_metric='logloss').fit(X, ybin)
    nz = y > 0
    reg = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, random_state=42,
                       n_jobs=-1).fit(X[nz], y[nz]) if nz.sum() > 20 else xgb
    models['rf'], models['xgb'] = rf, xgb
    models['hybrid'] = (clf, reg)
    return models

VINTAGES = {}
for year in (2021, 2022, 2023):
    cutoff = pd.Timestamp(f'{year}-12-01')
    VINTAGES[year] = {minlag: train_vintage(minlag, cutoff) for minlag in (4, 5)}
    print('vintage trained:', year, flush=True)

def ml_predict(model_name, target_month, j):
    minlag = 4 if j <= 3 else 5
    vy = min(target_month.year - 1, 2023)
    vy = max(vy, 2021)
    models = VINTAGES[vy][minlag]
    L = LONGS[minlag]
    rows = L[L.Date == target_month]
    if rows.empty: return None
    X = rows[FEATCOLS[minlag]].astype(float)
    if model_name == 'hybrid':
        clf, reg = models['hybrid']
        pred = clf.predict(X) * np.clip(reg.predict(X), 0, None)
    else:
        pred = np.clip(models[model_name].predict(X), 0, None)
    unitcols = [c for c in rows.columns if c.startswith('unit_')]
    out = {}
    for i, (_, r) in enumerate(rows.iterrows()):
        u = [c[5:] for c in unitcols if r[c] == 1][0]
        out[u] = float(pred[i])
    return out

# ---------- generate ----------
rows = []
ml_cache = {}
for t in ORIGINS:
    hist_end = t - pd.DateOffset(months=1)
    for j in range(5):
        m = t + pd.DateOffset(months=j)
        if m > MONTHS[-1]: continue
        for name in ('rf', 'xgb', 'hybrid'):
            key = (name, m, 4 if j <= 3 else 5, min(max(m.year - 1, 2021), 2023))
            if key not in ml_cache:
                ml_cache[key] = ml_predict(name, m, j)
            preds = ml_cache[key]
            if preds:
                for u in UNITS:
                    rows.append((u, t, m, j, name, preds.get(u, 0.0)))
    for u in UNITS:
        y = panel.loc[:hist_end, u].values.astype(float)
        cro = croston_variants(y)
        s = ses(1, y) if len(y) > 3 else float(np.mean(y))
        ar = arima_fc(y)
        for j in range(5):
            m = t + pd.DateOffset(months=j)
            if m > MONTHS[-1]: continue
            rows.append((u, t, m, j, 'mitra', mitra[u].get(str(m.year), mitra[u].get(m.year, 0)) / 12.0
                         if isinstance(mitra[u], dict) else 0.0))
            rows.append((u, t, m, j, 'naive', float(y[-1])))
            rows.append((u, t, m, j, 'ses', s))
            for k, v in cro.items():
                rows.append((u, t, m, j, k, v))
            rows.append((u, t, m, j, 'arima', float(ar[j])))
    print('origin done:', t.date(), flush=True)

fc = pd.DataFrame(rows, columns=['Unit', 'Origin', 'Month', 'h', 'Model', 'Forecast'])
act = panel.stack().rename('Actual').reset_index()
act.columns = ['Month', 'Unit', 'Actual']
fc = fc.merge(act, on=['Unit', 'Month'], how='left')
fc.to_parquet(os.path.join(HERE, 'forecasts.parquet'))
print('saved', fc.shape, 'models:', sorted(fc.Model.unique()))
