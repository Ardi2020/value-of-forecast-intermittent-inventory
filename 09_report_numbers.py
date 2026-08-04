"""Every number quoted in the manuscript, printed from the stored result tables.

Running this after a clean pipeline run is how the manuscript text is checked; each
block is labelled with the section that uses it.
"""
import os
import numpy as np
import pandas as pd
import simcore as S

HERE = os.path.dirname(os.path.abspath(__file__))
res = pd.read_csv(os.path.join(HERE, 'sim_results_audit.csv'))
acc = pd.read_csv(os.path.join(HERE, 'accuracy.csv'))
prof = pd.read_csv(os.path.join(HERE, 'intermittency_profile.csv'))
al = pd.read_csv(os.path.join(HERE, 'alignment_audit.csv'))
boot = pd.read_csv(os.path.join(HERE, 'bootstrap_intervals.csv'))
matched = pd.read_csv(os.path.join(HERE, 'matched_service.csv'))
PRIMARY = str(S.EVAL_END.date())


def grid(end=PRIMARY, k=0.0, ratio=10, sl=0.95, markup=-1):
    return res[(res.markup == markup) & (res.SL == sl) & (res.ratio == ratio) & res.lots
               & (res.term_k == k) & (res.eval_end == end)]


def summ(b):
    s = b.groupby('Model').total_cost.sum()
    idx = (100 * s / s['mitra']).round(1)
    fill = b.groupby('Model').apply(
        lambda g: (g.fill_rate * g.demand_2024).sum() / g.demand_2024.sum()).round(3)
    cyc = b.groupby('Model').cycle_service.mean().round(3)
    return pd.DataFrame({'cost_index': idx, 'agg_fill': fill,
                         'mean_cycle_service': cyc}).sort_values('cost_index')


print('§3  portfolio')
valid = sorted(grid().Unit.unique())
print('  valid units:', len(valid), '| segments:',
      prof[prof.Unit.isin(valid)].segment.value_counts().to_dict())
print('  units with no demand inside the primary window:',
      sorted(grid().loc[grid().demand_2024 == 0, 'Unit'].unique()))

print('\n§5.1  accuracy, medians over valid units, primary window')
print(acc.groupby('Model')[['MASE', 'RMSSE', 'MASE5', 'customMAPE']].median()
      .rename(index=S.NAME).sort_values('MASE').round(2).to_string())
best = acc.loc[acc.groupby(['Unit'])['customMAPE'].idxmin()].Model.value_counts()
print('  units where each method has the lowest adjusted MAPE:', best.to_dict())
bm = acc.loc[acc.groupby(['Unit'])['MASE'].idxmin()].Model.value_counts()
print('  units where each method has the lowest MASE:', bm.to_dict())

print('\n§5.2  cost at a COMMON z, primary window (complete horizon, Jan-Nov 2024)')
print(summ(grid()).rename(index=S.NAME).to_string())
print('\n  sensitivity, calendar year to December')
print(summ(grid(end='2024-12-01')).rename(index=S.NAME).to_string())
for k in (2.5, 5.0):
    s = grid(k=k).groupby('Model').total_cost.sum()
    print(f'  terminal charge k={k}: ', (100 * s / s['mitra']).round(1).to_dict())

print('\n§5.2  bootstrap over units, 1000 resamples')
print(boot.assign(Model=boot.Model.map(S.NAME)).to_string(index=False))

print('\n§5.3  common minimum achieved-fill requirement (primary cost comparison)')
print(matched.pivot(index='required_fill', columns='Model', values='cost_index')
      .rename(columns=S.NAME).to_string())
print('\n  achieved fill of the selected configuration')
print(matched.pivot(index='required_fill', columns='Model', values='achieved_fill')
      .rename(columns=S.NAME).to_string())
print('\n  achieved cycle service of the selected configuration')
print(matched.pivot(index='required_fill', columns='Model', values='achieved_cycle_service')
      .rename(columns=S.NAME).to_string())
print('\n  robustness: cost interpolated at exactly the required fill')
print(matched.pivot(index='required_fill', columns='Model', values='cost_index_exact_match')
      .rename(columns=S.NAME).to_string())

print('\n§5.4  current fixed mark-up versus an error-calibrated buffer')
cal = grid()
cal = cal[cal.Model == 'mitra']
den = cal.total_cost.sum()
for mk in (0.10, 0.20, 0.30):
    g = grid(markup=mk)
    g = g[g.Model == 'mitra']
    print(f'  mark-up {int(mk*100):>2}%: index {100*g.total_cost.sum()/den:6.1f}'
          f'  fill {(g.fill_rate*g.demand_2024).sum()/g.demand_2024.sum():.3f}'
          f'  cycle {g.cycle_service.mean():.3f}')
print(f'  calibrated : index  100.0  fill {(cal.fill_rate*cal.demand_2024).sum()/cal.demand_2024.sum():.3f}'
      f'  cycle {cal.cycle_service.mean():.3f}')

print('\n§5.5  cost index by SBC segment')
b2 = grid().merge(prof[['Unit', 'segment']], on='Unit')
seg = b2.groupby(['segment', 'Model']).total_cost.sum().unstack('Model')
print((100 * seg.div(seg['mitra'], axis=0)).round(1).rename(columns=S.NAME).to_string())

print('\n§5.6  RQ3 rank alignment')
print(al.groupby('metric').apply(lambda g: pd.Series({
    'median_rho': g.rho.median(),
    'volume_weighted_mean': np.average(g.rho.dropna(), weights=g.loc[g.rho.notna(), 'w'])
})).round(3).to_string())
big = prof[prof.Unit.isin(valid)].nlargest(2, 'mean_demand').Unit.tolist()
print('  two largest units', big, 'rho range:',
      round(al[al.Unit.isin(big)].rho.min(), 2), 'to', round(al[al.Unit.isin(big)].rho.max(), 2))

print('\n§5.7  robustness')
for ratio in (2, 5, 10, 20, 50):
    s = grid(ratio=ratio).groupby('Model').total_cost.sum()
    print(f'  b:h={ratio:>2}', (100 * s / s['mitra']).sort_values().round(1).head(4).to_dict())
for sl in (0.90, 0.95, 0.98):
    s = grid(sl=sl).groupby('Model').total_cost.sum()
    print(f'  SL={sl}', (100 * s / s['mitra']).sort_values().round(1).head(4).to_dict())
w = grid().groupby('Model').total_cost.sum()
wo = res[(res.markup == -1) & (res.SL == 0.95) & (res.ratio == 10) & ~res.lots
         & (res.term_k == 0) & (res.eval_end == PRIMARY)].groupby('Model').total_cost.sum()
print('  lot rounding cost ratio (with / without):', (w / wo).round(2).to_dict())
