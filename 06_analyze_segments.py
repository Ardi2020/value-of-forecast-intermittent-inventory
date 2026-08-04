"""Segmented + aggregate results (RQ1/RQ2) and refined RQ3."""
import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__))
res = pd.read_csv(os.path.join(HERE, 'sim_results.csv'))
acc = pd.read_csv(os.path.join(HERE, 'accuracy.csv'))
prof = pd.read_csv(os.path.join(HERE, 'intermittency_profile.csv'))

# SBC classification (Syntetos-Boylan-Croston cutoffs ADI 1.32, CV2 0.49)
def sbc(r):
    if r.ADI < 1.32:
        return 'smooth' if (r.CV2 < 0.49 or np.isnan(r.CV2)) else 'erratic'
    return 'intermittent' if (r.CV2 < 0.49 or np.isnan(r.CV2)) else 'lumpy'
prof['segment'] = prof.apply(sbc, axis=1)
print(prof[['Unit', 'pct_nonzero', 'ADI', 'CV2', 'mean_demand', 'segment']].sort_values('ADI').round(2).to_string())
prof.to_csv(os.path.join(HERE, 'intermittency_profile.csv'), index=False)

base = res[(res.SL == 0.95) & (res.ratio == 10) & res.lots].merge(prof[['Unit', 'segment', 'mean_demand']], on='Unit')

print('\n=== RQ1: TOTAL cost across all units (sum, normalized to mitra=100), fill = agg fill ===')
agg = base.groupby('Model').apply(lambda g: pd.Series({
    'total_cost': g.total_cost.sum(),
    'agg_fill': (g.fill_rate * g.demand_2024).sum() / g.demand_2024.sum(),
    'avg_oh_sum': g.avg_oh.sum(), 'avg_bl_sum': g.avg_bl.sum()}))
agg['cost_idx'] = 100 * agg.total_cost / agg.loc['mitra', 'total_cost']
print(agg.sort_values('cost_idx').round(3).to_string())

print('\n=== RQ2: cost index (mitra=100) per SBC segment ===')
seg = base.groupby(['segment', 'Model']).total_cost.sum().unstack('Model')
segidx = (100 * seg.div(seg['mitra'], axis=0)).round(1)
print(segidx.to_string())
print('\nsegment sizes:', prof.segment.value_counts().to_dict())

print('\n=== RQ1 robustness: cost index vs b:h ratio (all units, SL95, lots) ===')
for ratio in [2, 5, 10, 20, 50]:
    b = res[(res.SL == 0.95) & (res.ratio == ratio) & res.lots]
    s = b.groupby('Model').total_cost.sum()
    print(f'b:h={ratio:>2}', (100 * s / s['mitra']).sort_values().round(1).to_dict())

print('\n=== SL sensitivity (b:h=10, lots) ===')
for sl in [0.90, 0.95, 0.98]:
    b = res[(res.SL == sl) & (res.ratio == 10) & res.lots]
    s = b.groupby('Model').total_cost.sum()
    f = (b.fill_rate * b.demand_2024).sum() / b.demand_2024.sum()
    top3 = (100 * s / s['mitra']).sort_values().head(4).round(1).to_dict()
    print(f'SL={sl}', top3)

print('\n=== lot rounding effect (SL95 b:h=10): cost with lots / without ===')
w = res[(res.SL == 0.95) & (res.ratio == 10) & res.lots].groupby('Model').total_cost.sum()
wo = res[(res.SL == 0.95) & (res.ratio == 10) & ~res.lots].groupby('Model').total_cost.sum()
print((w / wo).round(2).to_string())

print('\n=== RQ3 refined: Spearman(accuracy rank, cost rank) weighted by unit size ===')
al = []
for u in prof.Unit:
    a = acc[acc.Unit == u].set_index('Model')
    c = base[base.Unit == u].set_index('Model')['total_cost']
    common = a.index.intersection(c.index)
    if len(common) < 5: continue
    wgt = prof.set_index('Unit').loc[u, 'mean_demand']
    for met in ['MASE', 'RMSSE', 'MASE5', 'customMAPE']:
        r, _ = spearmanr(a.loc[common, met], c.loc[common])
        al.append({'Unit': u, 'metric': met, 'rho': r, 'w': wgt,
                   'segment': prof.set_index('Unit').loc[u, 'segment']})
al = pd.DataFrame(al)
out = al.groupby('metric').apply(lambda g: pd.Series({
    'median_rho': g.rho.median(),
    'weighted_mean_rho': np.average(g.rho, weights=g.w)}))
print(out.round(3).to_string())
print('\nper segment (median rho):')
print(al.groupby(['segment', 'metric']).rho.median().unstack().round(3).to_string())
al.to_csv(os.path.join(HERE, 'alignment2.csv'), index=False)

# top-line per-model on the 5 largest units (economic core)
big = prof.nlargest(5, 'mean_demand').Unit.tolist()
bb = base[base.Unit.isin(big)]
s = bb.groupby('Model').total_cost.sum()
print('\n=== 5 largest units only: cost index (mitra=100) ===')
print((100 * s / s['mitra']).sort_values().round(1).to_string())
