"""B-03 / R3-B-13: compare cost under a COMMON MINIMUM ACHIEVED-SERVICE REQUIREMENT.

The order-up-to rule sets a buffer z * RMS(protection-interval error). Because that RMS
absorbs bias and intermittent errors are not normal, a common z does not deliver a common
achieved service: in the base case the annual plan reaches an aggregate fill of 0.988
while XGB-Tweedie reaches 0.981, so part of any cost difference is a service difference.

This script sweeps z per method and builds the aggregate service-cost frontier. Two
readings of that frontier are produced, and they are NOT the same thing:

  threshold   each method is charged the cost of its cheapest configuration achieving AT
              LEAST the required fill. This is a feasibility rule, not exact matching:
              the selected configurations land at different achieved fill rates, so the
              achieved fill and cycle service of each selected configuration are reported
              beside the cost and the residual difference stays visible. This is the
              comparison reported in the paper.

  exact       cost interpolated along the lower envelope at exactly the required fill.
              This removes the residual service difference at the price of interpolating
              between configurations that were never simulated, so it is a robustness
              check rather than the headline.

Writes service_frontier.csv (the whole sweep) and matched_service.csv (both readings,
with the achieved service of every selected configuration).
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
def selected(model, target):
    """Cheapest configuration of `model` achieving at least `target` aggregate fill.

    Taking the minimum over the feasible part of the sweep rather than interpolating
    avoids assuming that cost is monotone in z, which lot rounding breaks. The selected
    configuration generally lands ABOVE the requirement, which is why its achieved fill
    and cycle service are returned with it.
    """
    g = agg[(agg.Model == model) & (agg.agg_fill >= target - 1e-12)]
    if not len(g):
        return None
    return g.loc[g.total_cost.idxmin()]


def cost_exact(model, target):
    """Cost interpolated along the lower envelope at exactly `target` fill.

    The envelope is the running minimum cost as the achieved fill increases, so the
    interpolation runs between the cheapest configuration at or below the target and the
    cheapest at or above it.
    """
    g = agg[agg.Model == model].sort_values('agg_fill')
    lo = g[g.agg_fill <= target + 1e-12]
    hi = g[g.agg_fill >= target - 1e-12]
    if not len(hi):
        return np.nan
    if not len(lo):
        return float(hi.total_cost.min())
    a = lo.loc[lo.agg_fill.idxmax()]
    b = hi.loc[hi.agg_fill.idxmin()]
    if abs(b.agg_fill - a.agg_fill) < 1e-12:
        return float(min(a.total_cost, b.total_cost))
    w = (target - a.agg_fill) / (b.agg_fill - a.agg_fill)
    return float(a.total_cost + w * (b.total_cost - a.total_cost))


TARGETS = [0.95, 0.97, 0.98, 0.99]
out = []
for target in TARGETS:
    base = selected('mitra', target)
    base_exact = cost_exact('mitra', target)
    for mod in MODELS:
        sel = selected(mod, target)
        ce = cost_exact(mod, target)
        out.append({
            'required_fill': target, 'Model': mod,
            'cost_index': round(100 * sel.total_cost / base.total_cost, 1) if sel is not None else np.nan,
            'z_used': round(float(sel.z), 2) if sel is not None else np.nan,
            'achieved_fill': round(float(sel.agg_fill), 3) if sel is not None else np.nan,
            'achieved_cycle_service': round(float(sel.mean_cycle_service), 3) if sel is not None else np.nan,
            'cost_index_exact_match': round(100 * ce / base_exact, 1)
            if np.isfinite(ce) and np.isfinite(base_exact) else np.nan,
            'feasible': sel is not None})
matched = pd.DataFrame(out)
matched.to_csv(os.path.join(HERE, 'matched_service.csv'), index=False)

print('=== achieved aggregate fill by z ===')
piv = agg.pivot(index='z', columns='Model', values='agg_fill').round(3)
print(piv.rename(columns=S.NAME).to_string())
print()
print('=== cost index under a common MINIMUM fill requirement (partner annual = 100) ===')
print(matched.pivot(index='required_fill', columns='Model', values='cost_index')
      .rename(columns=S.NAME).to_string())
print()
print('=== achieved fill of the selected configuration (requirement is a floor) ===')
print(matched.pivot(index='required_fill', columns='Model', values='achieved_fill')
      .rename(columns=S.NAME).to_string())
print()
print('=== robustness: cost interpolated at EXACTLY the required fill ===')
print(matched.pivot(index='required_fill', columns='Model', values='cost_index_exact_match')
      .rename(columns=S.NAME).to_string())
print()
print('reachable fill range per method:')
print(agg.groupby('Model').agg_fill.agg(['min', 'max']).round(3).rename(index=S.NAME).to_string())
