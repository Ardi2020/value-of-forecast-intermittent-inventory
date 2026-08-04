"""Compare the two ML retraining schedules on the same policy.

R3-M-20: this script previously carried its own copy of the simulator. It now imports
simcore, like every other downstream script, and a regression check asserts that the
shared simulator reproduces the stored baseline grid for the same policy input.

Writes refit_frequency.csv.
"""
import itertools
import os
import numpy as np
import pandas as pd
import simcore as S

HERE = os.path.dirname(os.path.abspath(__file__))
panel, base_fc, months, reviews, units = S.load()
refit_ml = pd.read_parquet(os.path.join(HERE, 'forecasts_refit.parquet'))
FRAMES = {'yearly': base_fc,
          'every_origin': pd.concat(
              [base_fc[~base_fc.Model.isin(['rf', 'xgb', 'hybrid'])], refit_ml],
              ignore_index=True)}
ML = ['rf', 'xgb', 'hybrid']

scales = {}
for u in panel.columns:
    y = panel.loc[:'2023-12-01', u].values
    d = np.abs(np.diff(y))
    scales[u] = {'mae': max(d.mean(), 1e-9),
                 'rmse': max(np.sqrt((np.diff(y) ** 2).mean()), 1e-9)}


def accuracy(fc, models):
    ev = fc[(fc.Month >= str(S.EVAL_START.date())) & (fc.Month <= str(S.EVAL_END.date()))]
    out = []
    for (u, mod), g in ev.groupby(['Unit', 'Model']):
        if mod not in models or u not in units:
            continue
        e = g.Forecast - g.Actual
        g5 = g.groupby('Origin').agg(F5=('Forecast', 'sum'), A5=('Actual', 'sum'),
                                     n=('Forecast', 'size'))
        g5 = g5[g5.n == 5]
        apes = [100.0 if (a < .1 and f > .1) else (0.0 if a < .1 else abs((a - f) / a) * 100)
                for a, f in zip(g.Actual, g.Forecast)]
        out.append({'Unit': u, 'Model': mod,
                    'MASE': np.abs(e).mean() / scales[u]['mae'],
                    'RMSSE': np.sqrt((e ** 2).mean()) / scales[u]['rmse'],
                    'MASE5': (np.abs(g5.F5 - g5.A5).mean() / (scales[u]['mae'] * 5)) if len(g5) else np.nan,
                    'customMAPE': np.mean(apes)})
    return pd.DataFrame(out)


rows = []
for sched, fc in FRAMES.items():
    fsum = S.protection_errors(fc)
    sims = pd.DataFrame([S.simulate(panel, fc, fsum, months, reviews, u, mod, term_k=k)
                         for u, mod, k in itertools.product(units, ML + ['mitra'], [0.0, 2.5])])
    acc = accuracy(fc, ML)
    for k in (0.0, 2.5):
        s = sims[sims.term_k == k]
        tot = s.groupby('Model').total_cost.sum()
        for mod in ML:
            g = s[s.Model == mod]
            rows.append({'schedule': sched, 'term_k': k, 'Model': mod,
                         'cost_index': round(100 * tot[mod] / tot['mitra'], 1),
                         'agg_fill': round((g.fill_rate * g.demand_2024).sum() / g.demand_2024.sum(), 3),
                         'cycle_service_mean': round(g.cycle_service.mean(), 3),
                         'MASE_median': round(acc[acc.Model == mod].MASE.median(), 3),
                         'MASE5_median': round(acc[acc.Model == mod].MASE5.median(), 3),
                         'customMAPE_median': round(acc[acc.Model == mod].customMAPE.median(), 1)})

out = pd.DataFrame(rows)
out.to_csv(os.path.join(HERE, 'refit_frequency.csv'), index=False)
print(out.to_string(index=False))

# ---- regression check: the shared simulator reproduces the stored grid ----
grid = os.path.join(HERE, 'sim_results_audit.csv')
if os.path.exists(grid):
    ref = pd.read_csv(grid)
    ref = ref[(ref.markup == -1) & (ref.SL == 0.95) & (ref.ratio == 10) & ref.lots
              & (ref.term_k == 0) & (ref.eval_end == str(S.EVAL_END.date()))]
    fsum = S.protection_errors(base_fc)
    mine = pd.DataFrame([S.simulate(panel, base_fc, fsum, months, reviews, u, mod)
                         for u, mod in itertools.product(units, ML + ['mitra'])])
    chk = mine.merge(ref[['Unit', 'Model', 'total_cost']], on=['Unit', 'Model'],
                     suffixes=('', '_ref'))
    gap = (chk.total_cost - chk.total_cost_ref).abs().max()
    assert gap < 1e-6, f'shared simulator disagrees with the stored grid by {gap}'
    print(f'\nregression check passed on {len(chk)} unit-method pairs (max deviation {gap:.2e})')
