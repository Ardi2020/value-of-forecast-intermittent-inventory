"""M-03 (remaining part): sensitivity of the machine-learning results to refit frequency.

Baseline in the paper: yearly vintages -- a model trained on everything up to
31 December of year Y-1 produces every forecast issued during year Y.
Alternative here: refit at EVERY forecast origin on all demand observed up to
t-1, i.e. the most frequent retraining the information set allows.

Writes refit_frequency.csv (cost index and accuracy for both schedules) and
forecasts_refit.parquet (the per-origin-refit forecasts, for inspection).
"""
import pandas as pd, numpy as np, os, time, warnings, json, itertools
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
                       n_jobs=-1).fit(X[nz], y[nz], sample_weight=y[nz] ** 0.5) if nz.sum() > 20 else xgb
    return {'rf': rf, 'xgb': xgb, 'hybrid': (clf, reg)}


CACHE = os.path.join(HERE, 'forecasts_refit.parquet')
if os.path.exists(CACHE):
    ml = pd.read_parquet(CACHE)
    print('cached refit forecasts loaded', ml.shape, flush=True)
else:
    rows = []
    t0 = time.time()
    for oi, t in enumerate(ORIGINS):
        cutoff = t - pd.DateOffset(months=1)          # every observation available at t
        models_by_lag = {m: train_vintage(m, cutoff) for m in (4, 5)}
        for j in range(5):
            m = t + pd.DateOffset(months=j)
            if m > MONTHS[-1]:
                continue
            minlag = 4 if j <= 3 else 5
            L = LONGS[minlag]
            rr = L[L.Date == m]
            if rr.empty:
                continue
            X = rr[FEATCOLS[minlag]].astype(float)
            mods = models_by_lag[minlag]
            preds = {'rf': np.clip(mods['rf'].predict(X), 0, None),
                     'xgb': np.clip(mods['xgb'].predict(X), 0, None)}
            clf, reg = mods['hybrid']
            preds['hybrid'] = clf.predict_proba(X)[:, 1] * np.clip(reg.predict(X), 0, None)
            unitcols = [c for c in rr.columns if c.startswith('unit_')]
            for i, (_, r) in enumerate(rr.iterrows()):
                u = [c[5:] for c in unitcols if r[c] == 1][0]
                for name in ('rf', 'xgb', 'hybrid'):
                    rows.append((u, t, m, j, name, float(preds[name][i])))
        print(f'origin {t.date()}  ({oi+1}/{len(ORIGINS)})  {time.time()-t0:.0f}s', flush=True)
    ml = pd.DataFrame(rows, columns=['Unit', 'Origin', 'Month', 'h', 'Model', 'Forecast'])
    act = panel.stack().rename('Actual').reset_index()
    act.columns = ['Month', 'Unit', 'Actual']
    ml = ml.merge(act, on=['Unit', 'Month'], how='left')
    ml.to_parquet(CACHE)
    print('refit forecasts written', ml.shape, flush=True)
