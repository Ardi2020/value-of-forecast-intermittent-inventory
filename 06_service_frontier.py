"""B-03: compare cost at MATCHED achieved service instead of at a common target.

The order-up-to rule sets a buffer z * RMS(protection-interval error). Because that RMS
absorbs bias and intermittent errors are not normal, a common z does not deliver a common
achieved service: in the base case the annual plan reaches an aggregate fill of 0.988
while XGB-Tweedie reaches 0.981, so part of any cost difference is a service difference.

This script sweeps z per method, builds the aggregate service-cost frontier, and reads
off the cost each method needs to reach a common achieved fill rate. That comparison,
not the common-z one, is what a cost claim can rest on.

Writes service_frontier.csv (the whole sweep) and matched_service.csv (cost at the
matched fill targets).
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
ZGRID = sorted(set([round(x, 2) for x in np.arange(0.0, 3.01, 0.1)]
                   + [1.2816, 1.6449, 2.0537, 3.5, 4.0]))

rows = []
for u, mod, z in itertools.product(units, MODELS, ZGRID):
    rows.append(S.simulate(panel, fc, fsum, months, reviews, u, mod, z=z, ratio=10))
sweep = pd.DataFrame(rows)
sweep.to_csv(os.path.join(HERE, 'service_frontier.csv'), index=False)

agg = (sweep.groupby(['Model', 'z'])
       .apply(lambda g: pd.Series({
           'total_cost': g.total_cost.sum(),
           'agg_fill': (g.fill_rate * g.demand_2024).sum() / g.demand_2024.sum(),
           'mean_cycle_service': g.cycle_service.mean()}))
       .reset_index())

# cost index is always relative to the annual plan at ITS OWN operating point on the
# same matched-fill target, so the comparison is like for like
def cost_at_fill(model, target):
    """Cheapest configuration of `model` that achieves at least `target` aggregate fill.

    Taking the minimum over the feasible part of the sweep rather than interpolating
    avoids assuming that cost is monotone in z, which lot rounding breaks.
    """
    g = agg[(agg.Model == model) & (agg.agg_fill >= target - 1e-12)]
    return float(g.total_cost.min()) if len(g) else np.nan


def z_at_fill(model, target):
    g = agg[(agg.Model == model) & (agg.agg_fill >= target - 1e-12)]
    return float(g.loc[g.total_cost.idxmin(), 'z']) if len(g) else np.nan


TARGETS = [0.95, 0.97, 0.98, 0.99]
out = []
for target in TARGETS:
    base = cost_at_fill('mitra', target)
    for mod in MODELS:
        c = cost_at_fill(mod, target)
        out.append({'matched_fill': target, 'Model': mod,
                    'cost_index': round(100 * c / base, 1) if np.isfinite(c) and np.isfinite(base) else np.nan,
                    'z_used': z_at_fill(mod, target),
                    'feasible': bool(np.isfinite(c))})
matched = pd.DataFrame(out)
matched.to_csv(os.path.join(HERE, 'matched_service.csv'), index=False)

print('=== achieved aggregate fill by z (selected) ===')
piv = agg.pivot(index='z', columns='Model', values='agg_fill').round(3)
print(piv.rename(columns=S.NAME).to_string())
print()
print('=== cost index at MATCHED achieved fill (partner annual at the same fill = 100) ===')
print(matched.pivot(index='matched_fill', columns='Model', values='cost_index')
      .rename(columns=S.NAME).to_string())
print()
print('reachable fill range per method:')
print(agg.groupby('Model').agg_fill.agg(['min', 'max']).round(3).rename(index=S.NAME).to_string())
