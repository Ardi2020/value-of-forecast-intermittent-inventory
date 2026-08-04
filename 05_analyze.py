"""Accuracy metrics + accuracy->value alignment analysis (RQ1-RQ3)."""
import pandas as pd, numpy as np, os
HERE = os.path.dirname(os.path.abspath(__file__))
panel = pd.read_csv(os.path.join(HERE, 'unit_demand_monthly.csv'), index_col=0, parse_dates=True)
fc = pd.read_parquet(os.path.join(HERE, 'forecasts.parquet'))
res = pd.read_csv(os.path.join(HERE, 'sim_results.csv'))

EV = (fc.Month >= '2024-01-01') & (fc.Month <= '2024-12-01')
ev = fc[EV].copy()

# scale per unit: in-sample naive MAE / MSE on 2020-2023
scales = {}
for u in panel.columns:
    y = panel.loc[:'2023-12-01', u].values
    d = np.abs(np.diff(y))
    scales[u] = {'mae': max(d.mean(), 1e-9), 'rmse': max(np.sqrt((np.diff(y)**2).mean()), 1e-9)}

rows = []
for (u, mod), g in ev.groupby(['Unit', 'Model']):
    e = g.Forecast - g.Actual
    mase = np.abs(e).mean() / scales[u]['mae']
    rmsse = np.sqrt((e**2).mean()) / scales[u]['rmse']
    bias = e.mean()
    # 5-sum errors (Origin-level)
    g5 = g.groupby('Origin').agg(F5=('Forecast', 'sum'), A5=('Actual', 'sum'), n=('Forecast', 'size'))
    g5 = g5[g5.n == 5]
    mase5 = (np.abs(g5.F5 - g5.A5).mean() / (scales[u]['mae'] * 5)) if len(g5) else np.nan
    # TA-style custom MAPE for comparability
    apes = [100.0 if (a < .1 and f > .1) else (0.0 if a < .1 else abs((a - f) / a) * 100)
            for a, f in zip(g.Actual, g.Forecast)]
    rows.append({'Unit': u, 'Model': mod, 'MASE': mase, 'RMSSE': rmsse, 'Bias': bias,
                 'MASE5': mase5, 'customMAPE': np.mean(apes)})
acc = pd.DataFrame(rows)
acc.to_csv(os.path.join(HERE, 'accuracy.csv'), index=False)

# intermittency profile
prof = []
for u in panel.columns:
    y = panel.loc[:'2023-12-01', u]
    nz = y[y > 0]
    adi = len(y) / max(len(nz), 1)
    cv2 = (nz.std(ddof=1) / nz.mean())**2 if len(nz) > 2 else np.nan
    prof.append({'Unit': u, 'pct_nonzero': (y > 0).mean(), 'ADI': adi, 'CV2': cv2,
                 'mean_demand': y.mean()})
prof = pd.DataFrame(prof)
prof.to_csv(os.path.join(HERE, 'intermittency_profile.csv'), index=False)

# ---- headline tables ----
base = res[(res.SL == 0.95) & (res.ratio == 10) & res.lots]
merged = base.merge(acc, on=['Unit', 'Model']).merge(prof, on='Unit')
merged.to_csv(os.path.join(HERE, 'merged_unit_level.csv'), index=False)

# aggregate cost share-weighted (normalize cost per unit by that unit's mitra cost)
piv = base.pivot(index='Unit', columns='Model', values='total_cost')
rel = piv.div(piv['mitra'], axis=0)
print('=== relative total cost vs mitra (SL95, b:h=10, lots on; <1 = better) ===')
print(rel.median().sort_values().round(3).to_string())
print()
print('=== fill rate (median across units) ===')
print(base.pivot(index='Unit', columns='Model', values='fill_rate').median().sort_values(ascending=False).round(3).to_string())
print()
print('=== accuracy (median across units) ===')
print(acc.groupby('Model')[['MASE', 'RMSSE', 'MASE5', 'customMAPE']].median().round(3).to_string())

# RQ3: rank alignment accuracy metric vs cost, per unit
from scipy.stats import spearmanr
al = []
for u in panel.columns:
    a = acc[acc.Unit == u].set_index('Model')
    c = base[base.Unit == u].set_index('Model')['total_cost']
    common = a.index.intersection(c.index)
    if len(common) < 5: continue
    for met in ['MASE', 'RMSSE', 'MASE5', 'customMAPE']:
        r, _ = spearmanr(a.loc[common, met], c.loc[common])
        al.append({'Unit': u, 'metric': met, 'spearman_vs_cost': r})
al = pd.DataFrame(al)
al.to_csv(os.path.join(HERE, 'alignment.csv'), index=False)
print()
print('=== RQ3: median Spearman(metric rank, cost rank) per metric ===')
print(al.groupby('metric')['spearman_vs_cost'].median().round(3).to_string())
