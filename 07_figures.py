"""Publication figures (300 dpi, colorblind-safe Okabe-Ito subset, single-axis)."""
import pandas as pd, numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figures'); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25,
                     'grid.linewidth': 0.5, 'figure.dpi': 300})
BLUE, ORANGE, GREEN, GRAY = '#0072B2', '#E69F00', '#009E73', '#7F7F7F'

res = pd.read_csv(os.path.join(HERE, 'sim_results.csv'))
acc = pd.read_csv(os.path.join(HERE, 'accuracy.csv'))
prof = pd.read_csv(os.path.join(HERE, 'intermittency_profile.csv'))
base = res[(res.SL == 0.95) & (res.ratio == 10) & res.lots]
NAME = {'mitra': 'Partner annual', 'naive': 'Naive', 'ses': 'SES', 'croston': 'Croston',
        'sba': 'SBA', 'tsb': 'TSB', 'arima': 'ARIMA', 'rf': 'Random Forest',
        'xgb': 'XGBoost (Tweedie)', 'hybrid': 'Hybrid cls-reg'}

# ---- Fig 2: intermittency profile (SBC plane) ----
fig, ax = plt.subplots(figsize=(4.8, 3.6))
p = prof[prof.mean_demand > 0].copy()
sz = 12 + 300 * (p.mean_demand / p.mean_demand.max())
ax.scatter(p.ADI, p.CV2.fillna(0.02), s=sz, c=[BLUE if u.startswith('BR') else ORANGE for u in p.Unit],
           alpha=0.75, edgecolors='white', linewidths=1)
ax.axvline(1.32, color=GRAY, lw=0.8, ls='--'); ax.axhline(0.49, color=GRAY, lw=0.8, ls='--')
ax.set_xscale('log'); ax.set_xlabel('Average inter-demand interval, ADI (log scale)')
ax.set_ylabel('CV² of non-zero demand sizes')
for _, r in p.nlargest(4, 'mean_demand').iterrows():
    ax.annotate(r.Unit.replace('BR_', 'Broken ').replace('ST_', 'Stick '), (r.ADI, 0.02 if np.isnan(r.CV2) else r.CV2),
                textcoords='offset points', xytext=(6, 4), fontsize=7, color='#333')
ax.text(1.05, 1.02, 'smooth | erratic', transform=ax.get_xaxis_transform(), fontsize=7, color=GRAY)
ax.text(0.86, 0.95, 'Broken', color=BLUE, transform=ax.transAxes, fontsize=8)
ax.text(0.86, 0.88, 'Stick', color=ORANGE, transform=ax.transAxes, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig2_sbc_profile.png'), dpi=300); plt.close(fig)

# ---- Fig 3: RQ1 cost index bar ----
s = base.groupby('Model').total_cost.sum()
idx = (100 * s / s['mitra']).sort_values()
fill = base.groupby('Model').apply(lambda g: (g.fill_rate * g.demand_2024).sum() / g.demand_2024.sum())
fig, ax = plt.subplots(figsize=(5.4, 3.6))
cols = [GREEN if m == 'mitra' else (BLUE if m == 'xgb' else GRAY) for m in idx.index]
bars = ax.barh([NAME[m] for m in idx.index], idx.values, color=cols, height=0.62)
for b, v, m in zip(bars, idx.values, idx.index):
    ax.text(v + 4, b.get_y() + b.get_height() / 2, f'{v:.0f}  (fill {fill[m]:.2f})',
            va='center', fontsize=7.5, color='#333')
ax.axvline(100, color=GREEN, lw=0.9, ls=':')
ax.set_xlabel('Total inventory cost index (partner annual = 100)\nservice level 95%, backorder:holding = 10')
ax.set_xlim(0, idx.max() * 1.28); ax.invert_yaxis(); ax.grid(axis='y', alpha=0)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig3_cost_index.png'), dpi=300); plt.close(fig)

# ---- Fig 4: RQ2 heatmap segment x model ----
b2 = base.merge(prof[['Unit', 'segment']], on='Unit')
seg = b2.groupby(['segment', 'Model']).total_cost.sum().unstack('Model')
segidx = 100 * seg.div(seg['mitra'], axis=0)
order = ['mitra', 'xgb', 'ses', 'tsb', 'arima', 'sba', 'hybrid', 'rf', 'croston', 'naive']
segidx = segidx[order].reindex(['smooth', 'erratic', 'intermittent', 'lumpy'])
fig, ax = plt.subplots(figsize=(6.4, 2.6))
vals = segidx.values
import matplotlib.colors as mcolors
norm = mcolors.TwoSlopeNorm(vmin=60, vcenter=100, vmax=350)
im = ax.imshow(np.clip(vals, 60, 350), cmap='RdBu_r', norm=norm, aspect='auto')
ax.set_xticks(range(len(order)), [NAME[m] for m in order], rotation=35, ha='right', fontsize=7.5)
nunits = prof.segment.value_counts()
ax.set_yticks(range(4), [f'{s} (n={nunits.get(s, 0)})' for s in segidx.index], fontsize=8)
for i in range(vals.shape[0]):
    for j in range(vals.shape[1]):
        ax.text(j, i, f'{vals[i, j]:.0f}', ha='center', va='center', fontsize=7,
                color='white' if (vals[i, j] > 260 or vals[i, j] < 72) else 'black')
ax.set_title('Cost index by SBC segment (partner annual = 100; lower is better)', fontsize=9)
ax.grid(False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig4_segment_heatmap.png'), dpi=300); plt.close(fig)

# ---- Fig 5: RQ3 alignment dot plot ----
al = pd.read_csv(os.path.join(HERE, 'alignment2.csv'))
mets = ['customMAPE', 'MASE5', 'MASE', 'RMSSE']
lab = {'customMAPE': 'Adjusted MAPE', 'MASE': 'MASE (monthly)', 'MASE5': 'MASE (5-month sums)', 'RMSSE': 'RMSSE'}
fig, ax = plt.subplots(figsize=(5.0, 3.0))
for i, met in enumerate(mets):
    g = al[al.metric == met]
    ax.scatter(g.rho, np.full(len(g), i) + np.random.RandomState(1).uniform(-0.12, 0.12, len(g)),
               s=14, color=GRAY, alpha=0.55)
    med = g.rho.median()
    ax.scatter([med], [i], s=90, color=BLUE, zorder=3, marker='D')
    ax.text(med, i + 0.28, f'median {med:.2f}', ha='center', fontsize=7.5, color=BLUE)
ax.axvline(0, color='#444', lw=0.8)
ax.set_yticks(range(len(mets)), [lab[m] for m in mets], fontsize=8.5)
ax.set_xlabel("Spearman correlation between accuracy rank and\nsimulated cost rank (one dot per demand unit)")
ax.set_xlim(-1.05, 1.05)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig5_alignment.png'), dpi=300); plt.close(fig)

# ---- Fig 1: methodology flowchart ----
fig, ax = plt.subplots(figsize=(7.0, 2.9))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off'); ax.grid(False)
W, H = 0.205, 0.34
COL = [0.005, 0.270, 0.535, 0.795]
R1, R2 = 0.60, 0.06

def box(x, y, lines, fc='#EAF2FA', ec='#3C5A78'):
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle='round,pad=0,rounding_size=0.015',
                                fc=fc, ec=ec, lw=0.9))
    n = len(lines)
    step = H / (n + 0.6)
    top = y + H - step * 0.8
    for i, ln in enumerate(lines):
        ax.text(x + W / 2, top - i * step, ln, ha='center', va='center',
                fontsize=6.4, weight='bold' if i == 0 else 'normal', color='#111111')

def arrow(p1, p2, rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=7.5,
                                 color='#3C5A78', lw=0.85,
                                 connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0))

box(COL[0], R1, ['Demand data', '18 units, 55 months', 'FX + search interest'])
box(COL[1], R1, ['Rolling-origin forecasts', '10 methods, h = 1-5', 'origins 2022-2024'])
box(COL[2], R1, ['Accuracy evaluation', 'MASE, RMSSE,', 'adjusted MAPE'])
box(COL[1], R2, ['Safety stock per method', '\u03c3 from realised', '5-month-sum errors'])
box(COL[2], R2, ['(R,S) simulation', 'R = 2, L = 3, lot sizes', 'S = \u03a3F + z\u03c3'])
box(COL[3], (R1 + R2) / 2, ['Value assessment', 'cost index, fill rate,', 'segments, alignment'],
    fc='#E4F3EA', ec='#2F7A55')

M1, M2 = R1 + H / 2, R2 + H / 2
M3 = (R1 + R2) / 2 + H / 2
arrow((COL[0] + W, M1), (COL[1], M1))
arrow((COL[1] + W, M1), (COL[2], M1))
arrow((COL[1] + W / 2, R1), (COL[1] + W / 2, R2 + H))
arrow((COL[1] + W, M2), (COL[2], M2))
arrow((COL[2] + W, M1), (COL[3], M3 + 0.05), rad=-0.10)
arrow((COL[2] + W, M2), (COL[3], M3 - 0.05), rad=0.10)
fig.savefig(os.path.join(FIG, 'fig1_methodology.png'), dpi=300, bbox_inches='tight', pad_inches=0.03)
plt.close(fig)
print('saved fig1')
