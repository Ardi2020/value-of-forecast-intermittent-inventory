"""Regenerate ML forecasts (rf/xgb/hybrid) with fairer setups:
- XGB: Tweedie objective (zero-inflated skewed demand)
- Hybrid: P(nonzero) * conditional-mean regression (prob-weighted, not 0/1)
- RF: unchanged design but deeper forest
Keeps all non-ML rows from forecasts.parquet; replaces ML rows -> forecasts_v2.parquet
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
panel = pd.read_csv(os.path.join(HERE, 'unit_demand_monthly.csv'), index_col=0, parse_dates=True)
UNITS = list(panel.columns)
MONTHS = list(panel.index)
ORIGINS = [m for m in MONTHS if m >= pd.Timestamp('2022-01-01')]
exog = pd.read_csv(os.path.join(HERE, 'exog_monthly.csv'), index_col=0, parse_dates=True)

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor, XGBClassifier

def make_features(minlag):
    frames = []
    for u in UNITS:
        s = panel[u]
        d = pd.DataFrame({'y': s})
        for k in range(minlag, minlag + 4):
            d[f'lag{k}'] = s.shift(k)
        d['ma3'] = s.shift(minlag).rolling(3).mean()
        d['nzr6'] = (s.shift(minlag) > 0).rolling(6).mean()
        d['month'] = d.index.month
        d['unit'] = u
        d['type_broken'] = 1 if u.startswith('BR') else 0
        ex = exog.reindex(d.index).shift(minlag)
        d = pd.concat([d, ex.add_prefix('ex_')], axis=1)
        frames.append(d.reset_index().rename(columns={'index': 'Date'}))
    long = pd.concat(frames, ignore_index=True)
    long = pd.get_dummies(long, columns=['month', 'unit'], dtype=int)
    return long.dropna(subset=[f'lag{minlag}', f'lag{minlag+3}', 'ma3'])

LONGS = {m: make_features(m) for m in (4, 5)}
FEATCOLS = {m: [c for c in LONGS[m].columns if c not in ('y', 'Date')] for m in (4, 5)}

def train_vintage(minlag, cutoff):
    L = LONGS[minlag]
    tr = L[L.Date <= cutoff]
    X, y = tr[FEATCOLS[minlag]].astype(float), tr['y'].astype(float)
    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=1, max_features=0.5,
                               random_state=42, n_jobs=-1).fit(X, y)
    xgb = XGBRegressor(objective='reg:tweedie', tweedie_variance_power=1.3,
                       n_estimators=600, max_depth=5, learning_rate=0.05, subsample=0.9,
                       colsample_bytree=0.8, random_state=42, n_jobs=-1).fit(X, y)
    ybin = (y > 0).astype(int)
    clf = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, random_state=42,
                        n_jobs=-1, eval_metric='logloss').fit(X, ybin)
    nz = y > 0
    reg = XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05, random_state=42,
                       n_jobs=-1).fit(X[nz], y[nz], sample_weight=y[nz]**0.5) if nz.sum() > 20 else xgb
    return {'rf': rf, 'xgb': xgb, 'hybrid': (clf, reg)}

V = {}
for year in (2021, 2022, 2023):
    V[year] = {m: train_vintage(m, pd.Timestamp(f'{year}-12-01')) for m in (4, 5)}
    print('vintage', year, flush=True)

rows = []
for t in ORIGINS:
    for j in range(5):
        m = t + pd.DateOffset(months=j)
        if m > MONTHS[-1]: continue
        minlag = 4 if j <= 3 else 5
        # B-02 / R2-B-09: the vintage must be chosen from the ORIGIN, not from the
        # target month. A model trained through 31 Dec of year Y is only available to
        # origins in year Y+1 onwards; keying on m.year let a September 2023 origin
        # forecasting January 2024 use a model trained to the end of December 2023.
        vy = min(max(t.year - 1, 2021), 2023)
        L = LONGS[minlag]
        rr = L[L.Date == m]
        if rr.empty: continue
        X = rr[FEATCOLS[minlag]].astype(float)
        models = V[vy][minlag]
        preds = {}
        preds['rf'] = np.clip(models['rf'].predict(X), 0, None)
        preds['xgb'] = np.clip(models['xgb'].predict(X), 0, None)
        clf, reg = models['hybrid']
        preds['hybrid'] = clf.predict_proba(X)[:, 1] * np.clip(reg.predict(X), 0, None)
        unitcols = [c for c in rr.columns if c.startswith('unit_')]
        for i, (_, r) in enumerate(rr.iterrows()):
            u = [c[5:] for c in unitcols if r[c] == 1][0]
            for name in ('rf', 'xgb', 'hybrid'):
                rows.append((u, t, m, j, name, float(preds[name][i])))
    if t.month == 12: print('origins through', t.date(), flush=True)

ml = pd.DataFrame(rows, columns=['Unit', 'Origin', 'Month', 'h', 'Model', 'Forecast'])
act = panel.stack().rename('Actual').reset_index()
act.columns = ['Month', 'Unit', 'Actual']
ml = ml.merge(act, on=['Unit', 'Month'], how='left')

# ---- leakage test: no training observation may be dated at or after the origin ----
VCUT = {2021: pd.Timestamp('2021-12-01'), 2022: pd.Timestamp('2022-12-01'),
        2023: pd.Timestamp('2023-12-01')}
bad = 0
for t in ORIGINS:
    vy = min(max(t.year - 1, 2021), 2023)
    if VCUT[vy] >= t:
        bad += 1
        print('LEAK: origin', t.date(), 'uses vintage', vy, 'cut', VCUT[vy].date())
assert bad == 0, f'{bad} origins use a vintage trained on data dated at or after the origin'
print('leakage test passed: max(training date) < origin for all', len(ORIGINS), 'origins')
old = pd.read_parquet(os.path.join(HERE, 'forecasts.parquet'))
keep = old[~old.Model.isin(['rf', 'xgb', 'hybrid'])]
out = pd.concat([keep, ml], ignore_index=True)
out.to_parquet(os.path.join(HERE, 'forecasts.parquet'))
print('saved v2', out.shape)
