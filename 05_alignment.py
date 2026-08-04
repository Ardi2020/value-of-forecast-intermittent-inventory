"""Accuracy metrics, intermittency profile, and accuracy-to-cost rank alignment.

Released because the Round-2 audit found no script generating alignment_audit.csv.

The accuracy window matches the primary cost window (January-November 2024) so that the
rank correlation in RQ3 compares two orderings computed over the same months. The
calendar-year variant is written alongside it as a sensitivity.

Writes accuracy.csv, accuracy_calendar.csv, intermittency_profile.csv,
alignment_audit.csv.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
import simcore as S

HERE = os.path.dirname(os.path.abspath(__file__))
panel, fc, months, reviews, units = S.load()
res = pd.read_csv(os.path.join(HERE, 'sim_results_audit.csv'))

# in-sample scaling denominators, estimated on the pre-2024 history only
scales = {}
for u in panel.columns:
    y = panel.loc[:'2023-12-01', u].values
    d = np.abs(np.diff(y))
    scales[u] = {'mae': max(d.mean(), 1e-9),
                 'rmse': max(np.sqrt((np.diff(y) ** 2).mean()), 1e-9)}


def accuracy(end):
    ev = fc[(fc.Month >= '2024-01-01') & (fc.Month <= end)]
    rows = []
    for (u, mod), g in ev.groupby(['Unit', 'Model']):
        if u not in units:
            continue
        e = g.Forecast - g.Actual
        g5 = g.groupby('Origin').agg(F5=('Forecast', 'sum'), A5=('Actual', 'sum'),
                                     n=('Forecast', 'size'))
        g5 = g5[g5.n == 5]
        apes = [100.0 if (a < .1 and f > .1) else (0.0 if a < .1 else abs((a - f) / a) * 100)
                for a, f in zip(g.Actual, g.Forecast)]
        rows.append({'Unit': u, 'Model': mod,
                     'MASE': np.abs(e).mean() / scales[u]['mae'],
                     'RMSSE': np.sqrt((e ** 2).mean()) / scales[u]['rmse'],
                     'Bias': e.mean(),
                     'MASE5': (np.abs(g5.F5 - g5.A5).mean() / (scales[u]['mae'] * 5)) if len(g5) else np.nan,
                     'customMAPE': np.mean(apes)})
    return pd.DataFrame(rows)


acc = accuracy('2024-11-01')
acc.to_csv(os.path.join(HERE, 'accuracy.csv'), index=False)
accuracy('2024-12-01').to_csv(os.path.join(HERE, 'accuracy_calendar.csv'), index=False)

prof = []
for u in panel.columns:
    y = panel.loc[:'2023-12-01', u]
    nz = y[y > 0]
    adi = len(y) / max(len(nz), 1)
    cv2 = (nz.std(ddof=1) / nz.mean()) ** 2 if len(nz) > 2 else np.nan
    prof.append({'Unit': u, 'pct_nonzero': (y > 0).mean(), 'ADI': adi, 'CV2': cv2,
                 'mean_demand': y.mean()})
prof = pd.DataFrame(prof)


def sbc(r):
    if r.ADI < 1.32:
        return 'smooth' if (r.CV2 < 0.49 or np.isnan(r.CV2)) else 'erratic'
    return 'intermittent' if (r.CV2 < 0.49 or np.isnan(r.CV2)) else 'lumpy'


prof['segment'] = prof.apply(sbc, axis=1)
prof.to_csv(os.path.join(HERE, 'intermittency_profile.csv'), index=False)

# ---- RQ3: within-unit rank correlation, average ranks for ties ----
base = res[(res.markup == -1) & (res.SL == 0.95) & (res.ratio == 10) & res.lots
           & (res.term_k == 0) & (res.eval_end == str(S.EVAL_END.date()))]
al = []
for u in units:
    a = acc[acc.Unit == u].set_index('Model')
    c = base[base.Unit == u].set_index('Model')['total_cost']
    common = a.index.intersection(c.index)
    if len(common) < 5:
        continue
    cr = rankdata(c.loc[common].values)
    if len(np.unique(cr)) < 2:
        continue
    w = float(prof.set_index('Unit').loc[u, 'mean_demand'])
    seg = prof.set_index('Unit').loc[u, 'segment']
    for met in ['MASE', 'RMSSE', 'MASE5', 'customMAPE']:
        ar = rankdata(a.loc[common, met].values)
        rho = np.nan if len(np.unique(ar)) < 2 else spearmanr(ar, cr).statistic
        al.append({'Unit': u, 'metric': met, 'rho': rho, 'w': w, 'segment': seg})
al = pd.DataFrame(al)
al.to_csv(os.path.join(HERE, 'alignment_audit.csv'), index=False)

print('units with a defined correlation:', al.Unit.nunique(), 'of', len(units))
print('\nmedian accuracy across valid units (primary window):')
print(acc.groupby('Model')[['MASE', 'RMSSE', 'MASE5', 'customMAPE']].median().round(3)
      .rename(index=S.NAME).sort_values('MASE').to_string())
print('\nRQ3 rank correlation with simulated cost:')
print(al.groupby('metric').apply(lambda g: pd.Series({
    'median_rho': g.rho.median(),
    'volume_weighted_mean_rho': np.average(g.rho.dropna(),
                                           weights=g.loc[g.rho.notna(), 'w'])})).round(3).to_string())
