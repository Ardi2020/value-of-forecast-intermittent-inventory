"""B-07 / R2-B-10: released implementation of the uncertainty interval.

Estimand. The reported quantity is the portfolio cost index of a method relative to the
annual plan, where the portfolio is the set of demand units the company actually
operates. The sixteen units are that portfolio, not a sample from a larger population,
so the interval below is a sensitivity to portfolio COMPOSITION: it answers "how much of
the observed ranking depends on which units happen to be in this portfolio", not "what
would happen in another company".

Resampling unit. One demand unit, with its entire time path kept intact, so the
within-unit dependence across months is preserved. Sampling is with replacement to the
original portfolio size.

Two indices are bootstrapped:
  * the common-z index at the base case (z from a 95% target), and
  * the index under a common minimum achieved-fill requirement, built by
    service_frontier.py, which is the comparison the cost claim rests on.

An interval that spans 100 means the data do not resolve the direction of the
difference. It is NOT evidence of equivalence; no equivalence margin was specified in
advance and none is claimed.

Writes bootstrap_intervals.csv.
"""
import os
import numpy as np
import pandas as pd
import simcore as S

HERE = os.path.dirname(os.path.abspath(__file__))
NBOOT = 1000
SEED = 20260804          # fixed so the interval is reproducible
MATCH_FILL = 0.98

res = pd.read_csv(os.path.join(HERE, 'sim_results_audit.csv'))
base = res[(res.markup == -1) & (res.SL == 0.95) & (res.ratio == 10) & res.lots
           & (res.term_k == 0) & (res.eval_end == str(S.EVAL_END.date()))]
cost = base.pivot(index='Unit', columns='Model', values='total_cost')
UNITS = list(cost.index)
MODELS = list(cost.columns)

sweep = pd.read_csv(os.path.join(HERE, 'service_frontier.csv'))
ZS = sorted(sweep.z.unique())
cost_z = {m: sweep[sweep.Model == m].pivot(index='Unit', columns='z', values='total_cost')
          .reindex(UNITS)[ZS].values for m in MODELS}
# four units record no demand inside the primary window; their fill rate is undefined and
# they contribute nothing to either side of the aggregate ratio, but they do contribute cost
served_z = {m: np.nan_to_num(sweep[sweep.Model == m]
                             .pivot(index='Unit', columns='z', values='fill_rate')
                             .reindex(UNITS)[ZS].values
                             * sweep[sweep.Model == m]
                             .pivot(index='Unit', columns='z', values='demand_2024')
                             .reindex(UNITS)[ZS].values) for m in MODELS}
dem = (sweep[sweep.Model == MODELS[0]].pivot(index='Unit', columns='z', values='demand_2024')
       .reindex(UNITS)[ZS].values[:, 0])


def matched_cost(model, idx, target):
    """Cheapest configuration of `model` on the resampled portfolio reaching `target`."""
    c = cost_z[model][idx].sum(axis=0)                       # total cost per z
    D = dem[idx].sum()
    if D <= 0:
        return np.nan
    f = served_z[model][idx].sum(axis=0) / D
    ok = f >= target - 1e-12
    return float(c[ok].min()) if ok.any() else np.nan


rng = np.random.default_rng(SEED)
common, matched = {m: [] for m in MODELS}, {m: [] for m in MODELS}
n = len(UNITS)
for _ in range(NBOOT):
    idx = rng.integers(0, n, n)
    tot = cost.values[idx].sum(axis=0)
    denom = tot[MODELS.index('mitra')]
    for j, m in enumerate(MODELS):
        common[m].append(100 * tot[j] / denom)
    base_m = matched_cost('mitra', idx, MATCH_FILL)
    for m in MODELS:
        cm = matched_cost(m, idx, MATCH_FILL)
        matched[m].append(100 * cm / base_m if np.isfinite(cm) and np.isfinite(base_m) else np.nan)

point_common = 100 * cost.sum() / cost.sum()['mitra']
rows = []
for m in MODELS:
    a = np.array(common[m])
    b = np.array(matched[m])
    b = b[np.isfinite(b)]
    rows.append({'Model': m, 'basis': 'common z (95% target)',
                 'point': round(point_common[m], 1),
                 'lo': round(np.percentile(a, 2.5), 1), 'hi': round(np.percentile(a, 97.5), 1),
                 'spans_100': bool(np.percentile(a, 2.5) < 100 < np.percentile(a, 97.5)),
                 'n_boot': NBOOT})
    rows.append({'Model': m, 'basis': f'minimum fill requirement {MATCH_FILL}',
                 'point': np.nan, 'lo': round(np.percentile(b, 2.5), 1),
                 'hi': round(np.percentile(b, 97.5), 1),
                 'spans_100': bool(np.percentile(b, 2.5) < 100 < np.percentile(b, 97.5)),
                 'n_boot': len(b)})
out = pd.DataFrame(rows)
matched_point = pd.read_csv(os.path.join(HERE, 'matched_service.csv'))
mp = matched_point[matched_point.required_fill == MATCH_FILL].set_index('Model').cost_index
out.loc[out.basis.str.startswith('minimum'), 'point'] = \
    out.loc[out.basis.str.startswith('minimum'), 'Model'].map(mp).values
out.to_csv(os.path.join(HERE, 'bootstrap_intervals.csv'), index=False)
print(out.assign(Model=out.Model.map(S.NAME)).to_string(index=False))
