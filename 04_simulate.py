"""Full simulation grid, built on the single simulator in simcore.py.

Replaces the earlier 04_simulate.py, which carried its own copy of the policy logic.

Grid:
  1. error-calibrated buffer x {90, 95, 98}% target x {2, 5, 10, 20, 50} backorder ratio,
     plus one no-lot-rounding run per target, over the primary window;
  2. terminal charges k in {2.5, 5} and the calendar-year window, at the base case;
  3. the company's current policy: annual plan with a fixed mark-up of 10 / 20 / 30 %.

Writes sim_results_audit.csv and error_decomposition.csv.
"""
import itertools
import os
import numpy as np
import pandas as pd
import simcore as S

HERE = os.path.dirname(os.path.abspath(__file__))
panel, fc, months, reviews, units = S.load()
fsum = S.protection_errors(fc)
MODELS = sorted(fc.Model.unique())


def sim(u, mod, **kw):
    return S.simulate(panel, fc, fsum, months, reviews, u, mod, **kw)


rows = []
for u, mod, sl in itertools.product(units, MODELS, [0.90, 0.95, 0.98]):
    for ratio in [2, 5, 10, 20, 50]:
        rows.append(sim(u, mod, sl=sl, ratio=ratio))
    rows.append(sim(u, mod, sl=sl, ratio=10, lots=False))
for u, mod in itertools.product(units, MODELS):
    for k in [2.5, 5.0]:
        rows.append(sim(u, mod, sl=0.95, ratio=10, term_k=k))
    rows.append(sim(u, mod, sl=0.95, ratio=10, eval_end=S.EVAL_END_CALENDAR))
for u, mk in itertools.product(units, [0.10, 0.20, 0.30]):
    for ratio in [2, 5, 10, 20, 50]:
        rows.append(sim(u, 'mitra', sl=0.95, ratio=ratio, markup=mk))
    for k in [2.5, 5.0]:
        rows.append(sim(u, 'mitra', sl=0.95, ratio=10, markup=mk, term_k=k))
    rows.append(sim(u, 'mitra', sl=0.95, ratio=10, markup=mk, eval_end=S.EVAL_END_CALENDAR))

res = pd.DataFrame(rows)
res.to_csv(os.path.join(HERE, 'sim_results_audit.csv'), index=False)
print('rows:', len(res), '| units:', len(units), '| primary window ends', S.EVAL_END.date())

dec = []
for u in units:
    for mod in MODELS:
        e = fsum[(fsum.Unit == u) & (fsum.Model == mod)
                 & (fsum.Origin >= '2024-01-01') & (fsum.Origin <= '2024-11-01')]['err']
        if len(e) < 3:
            continue
        dec.append({'Unit': u, 'Model': mod, 'bias': e.mean(), 'sd': e.std(ddof=1),
                    'rms': np.sqrt((e ** 2).mean()), 'n': len(e)})
pd.DataFrame(dec).to_csv(os.path.join(HERE, 'error_decomposition.csv'), index=False)
print('error decomposition written')
