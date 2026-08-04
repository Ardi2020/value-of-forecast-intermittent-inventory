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
                     'grid.linewidth': 0.5, 'figure.dpi': 600})
BLUE, ORANGE, GREEN, GRAY = '#0072B2', '#E69F00', '#009E73', '#7F7F7F'

res = pd.read_csv(os.path.join(HERE, 'sim_results_audit.csv'))
acc = pd.read_csv(os.path.join(HERE, 'accuracy.csv'))
prof = pd.read_csv(os.path.join(HERE, 'intermittency_profile.csv'))
# base case: error-calibrated buffer, 95% target, b:h = 10, lot rounding, no terminal charge,
# primary complete-horizon window (January-November 2024); all-zero units are already
# absent from sim_results_audit.csv
base = res[(res.markup == -1) & (res.SL == 0.95) & (res.ratio == 10) & res.lots
           & (res.term_k == 0) & (res.eval_end == '2024-11-01')]
VALID = sorted(base.Unit.unique())
prof = prof[prof.Unit.isin(VALID)].copy()
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
ax.set_ylim(-0.08, 1.04)
# quadrant names, placed in empty space rather than on the cut-off lines
ax.text(1.025, 0.93, 'erratic', rotation=90, ha='center', va='center', fontsize=6.5, color=GRAY)
ax.text(1.025, 0.14, 'smooth', rotation=90, ha='center', va='center', fontsize=6.5, color=GRAY)
ax.text(17, 0.86, 'lumpy', fontsize=6.5, color=GRAY)
ax.text(17, 0.32, 'intermittent', fontsize=6.5, color=GRAY)
ax.text(0.86, 0.62, 'Broken', color=BLUE, transform=ax.transAxes, fontsize=8)
ax.text(0.86, 0.55, 'Stick', color=ORANGE, transform=ax.transAxes, fontsize=8)

# labels for the four largest units: try candidate offsets and keep the first that
# collides with no marker, no other label and no axes edge
fig.canvas.draw()
REND = fig.canvas.get_renderer()
pts = [(ax.transData.transform((r.ADI, 0.02 if np.isnan(r.CV2) else r.CV2)),
        np.sqrt(s / np.pi) * fig.dpi / 72 + 2)
       for (_, r), s in zip(p.iterrows(), sz)]
placed = [t.get_window_extent(REND) for t in ax.texts]
AXBB = ax.get_window_extent(REND)
CAND = [(9, 5, 'left'), (9, -11, 'left'), (-9, 5, 'right'), (-9, -11, 'right'),
        (9, 16, 'left'), (9, -22, 'left'), (-9, 16, 'right'), (-9, -22, 'right')]
for _, r in p.nlargest(4, 'mean_demand').iterrows():
    y = 0.02 if np.isnan(r.CV2) else r.CV2
    label = r.Unit.replace('BR_', 'Broken ').replace('ST_', 'Stick ')
    for dx, dy, ha in CAND:
        t = ax.annotate(label, (r.ADI, y), textcoords='offset points', xytext=(dx, dy),
                        ha=ha, fontsize=7, color='#333')
        bb = t.get_window_extent(REND)
        hits = any(bb.overlaps(b) for b in placed) or not AXBB.containsx(bb.x0) \
            or not AXBB.containsx(bb.x1) or not AXBB.containsy(bb.y1)
        if not hits:
            for (cx, cy), rad in pts:
                nx, ny = min(max(cx, bb.x0), bb.x1), min(max(cy, bb.y0), bb.y1)
                if (nx - cx) ** 2 + (ny - cy) ** 2 < rad ** 2:
                    hits = True
                    break
        if not hits:
            placed.append(bb)
            break
        t.remove()
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig2_sbc_profile.png'), dpi=600); plt.close(fig)

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
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig3_cost_index.png'), dpi=600); plt.close(fig)

# ---- Fig 3b: service-cost frontier (B-03) ----
sw = pd.read_csv(os.path.join(HERE, 'service_frontier.csv'))
fr = (sw.groupby(['Model', 'z'])
      .apply(lambda g: pd.Series({
          'cost': g.total_cost.sum(),
          'fill': (g.fill_rate * g.demand_2024).sum() / g.demand_2024.sum()}))
      .reset_index())
denom = fr[(fr.Model == 'mitra')].cost.min()
fr['idx'] = 100 * fr.cost / denom
HILITE = {'mitra': (GREEN, 'Partner annual', 2.0), 'xgb': (BLUE, 'XGBoost (Tweedie)', 2.0),
          'ses': (ORANGE, 'SES', 1.4)}
fig, ax = plt.subplots(figsize=(5.4, 3.6))
for mod, g in fr.groupby('Model'):
    g = g.sort_values('z')
    g = g[(g.fill >= 0.85) & (g.fill <= 1.0)]
    if mod in HILITE:
        c, lab, lw = HILITE[mod]
        ax.plot(g.fill, g.idx, color=c, lw=lw, label=lab, zorder=3)
    else:
        ax.plot(g.fill, g.idx, color=GRAY, lw=0.8, alpha=0.55, zorder=1)
ax.annotate('seven other methods', (0.905, 205), fontsize=7.5, color=GRAY)
for tgt in (0.95, 0.98):
    ax.axvline(tgt, color='#bbbbbb', lw=0.7, ls=':', zorder=0)
ax.set_xlabel('Achieved aggregate fill rate')
ax.set_ylabel('Total inventory cost index\n(cheapest partner-annual policy = 100)')
ax.set_xlim(0.90, 1.002)
ax.set_ylim(80, 300)
ax.legend(fontsize=7.5, frameon=False, loc='upper left')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig4_service_frontier.png'), dpi=600); plt.close(fig)

# ---- Fig 5: RQ2 heatmap segment x model ----
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
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig5_segment_heatmap.png'), dpi=600); plt.close(fig)

# ---- Fig 6: RQ3 alignment dot plot ----
al = pd.read_csv(os.path.join(HERE, 'alignment_audit.csv'))
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
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig6_alignment.png'), dpi=600); plt.close(fig)

# ---- Fig 1: methodology flowchart ----
fig, ax = plt.subplots(figsize=(7.0, 2.8))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off'); ax.grid(False)
W, H = 0.205, 0.34
COL = [0.005, 0.270, 0.535, 0.795]
R1, R2 = 0.60, 0.06
fig.canvas.draw()
REND = fig.canvas.get_renderer()

def box_width_px():
    p0 = ax.transData.transform((0, 0)); p1 = ax.transData.transform((W, 0))
    return p1[0] - p0[0]

MAXPX = box_width_px() * 0.86          # keep a visible margin inside the frame

def fitted(x, y, s, bold):
    """place text, shrinking the font until it fits inside the box"""
    size = 7.0
    while size > 4.0:
        t = ax.text(x, y, s, ha='center', va='center', fontsize=size,
                    weight='bold' if bold else 'normal', color='#111111')
        if t.get_window_extent(REND).width <= MAXPX:
            return t
        t.remove(); size -= 0.2
    return ax.text(x, y, s, ha='center', va='center', fontsize=4.0,
                   weight='bold' if bold else 'normal', color='#111111')

def box(x, y, lines, fc='#EAF2FA', ec='#3C5A78'):
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle='round,pad=0,rounding_size=0.015',
                                fc=fc, ec=ec, lw=0.9))
    n = len(lines)
    step = H / (n + 0.7)
    top = y + H - step * 0.85
    for i, ln in enumerate(lines):
        fitted(x + W / 2, top - i * step, ln, bold=(i == 0))

def arrow(p1, p2, rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=7.5,
                                 color='#3C5A78', lw=0.85,
                                 connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0))

box(COL[0], R1, ['Demand data', '16 units, 55 months', 'FX + search interest'])
box(COL[1], R1, ['Forecasting', '10 methods, rolling', 'origins, h = 1-5'])
box(COL[2], R1, ['Accuracy metrics', 'MASE, RMSSE,', 'adjusted MAPE'])
box(COL[1], R2, ['Safety stock', '\u03c3 from realised', '5-month-sum errors'])
box(COL[2], R2, ['(R,S) simulation', 'R = 2, L = 3, lots', 'S = \u03a3F + z\u03c3'])
box(COL[3], (R1 + R2) / 2, ['Value assessment', 'cost, fill rate,', 'segments, alignment'],
    fc='#E4F3EA', ec='#2F7A55')

M1, M2 = R1 + H / 2, R2 + H / 2
M3 = (R1 + R2) / 2 + H / 2
arrow((COL[0] + W, M1), (COL[1], M1))
arrow((COL[1] + W, M1), (COL[2], M1))
arrow((COL[1] + W / 2, R1), (COL[1] + W / 2, R2 + H))
arrow((COL[1] + W, M2), (COL[2], M2))
arrow((COL[2] + W, M1), (COL[3], M3 + 0.05), rad=-0.10)
arrow((COL[2] + W, M2), (COL[3], M3 - 0.05), rad=0.10)
fig.savefig(os.path.join(FIG, 'fig1_methodology.png'), dpi=600, bbox_inches='tight', pad_inches=0.03)
plt.close(fig)
print('saved fig1 (auto-fitted text)')
