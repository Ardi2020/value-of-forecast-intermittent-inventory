"""Sequential (R,S) simulation per unit x model.
Review every R=2 months (reviews at Jan/Mar/... starting 2023-01 warm-up;
evaluation window 2024-01..2024-12). Lead time L=3 months. Protection = L+R = 5.
S_t = sum of forecasts over months t..t+4 + z * sigma_t, where sigma_t = std of
past realized 5-month-sum forecast errors for that (unit, model), using origins
whose outcome windows completed before t (no leakage). Fallback sigma: std of
5-month rolling demand sums over history * 1.0.
Order-up-to: order = max(0, S - IP); IP = on_hand + on_order - backorder.
Lot rounding scenario: Broken -> ceil to 25,025 kg (25,000 for the mixed FR unit
uses 25,025 too — documented simplification); Stick -> ceil to 20 kg.
Backorders carried; costs: h=1/kg/month on ending on-hand, b = ratio * h on ending backorder.
"""
import pandas as pd, numpy as np, os, itertools, json
HERE = os.path.dirname(os.path.abspath(__file__))
panel = pd.read_csv(os.path.join(HERE, 'unit_demand_monthly.csv'), index_col=0, parse_dates=True)
fc = pd.read_parquet(os.path.join(HERE, 'forecasts.parquet'))
MONTHS = list(panel.index)
L, R = 3, 2
Z = {0.90: 1.2816, 0.95: 1.6449, 0.98: 2.0537}
REVIEWS = [m for m in MONTHS if m >= pd.Timestamp('2023-01-01') and ((m.year - 2023) * 12 + m.month - 1) % R == 0]
EVAL_START, EVAL_END = pd.Timestamp('2024-01-01'), pd.Timestamp('2024-12-01')
MODELS = sorted(fc.Model.unique())
UNITS = list(panel.columns)

# 5-month-sum forecast (and error history) per unit/model/origin
fsum = (fc.groupby(['Unit', 'Model', 'Origin'])
          .agg(F5=('Forecast', 'sum'), A5=('Actual', 'sum'), n=('Forecast', 'size'))
          .reset_index())
fsum = fsum[fsum.n == 5]
fsum['err'] = fsum.F5 - fsum.A5

def sigma_at(u, mod, t):
    """std of past errors from origins o with o+4 <= t-1 (window fully realized)."""
    cut = t - pd.DateOffset(months=5)
    e = fsum[(fsum.Unit == u) & (fsum.Model == mod) & (fsum.Origin <= cut)]['err'].tail(12)
    if len(e) >= 6:
        return float(np.sqrt((e**2).mean()))  # RMS of recent errors (bias-inclusive)
    hist = panel.loc[:t - pd.DateOffset(months=1), u].rolling(5).sum().dropna()
    return float(hist.std(ddof=1)) if len(hist) > 3 else float(panel[u].std())

def lot_round(qty, unit):
    if qty <= 0: return 0.0
    base = 25025.0 if unit.startswith('BR') else 20.0
    return float(np.ceil(qty / base) * base)

def simulate(u, mod, sl, ratio, lots=True):
    fu = fc[(fc.Unit == u) & (fc.Model == mod)].set_index(['Origin', 'Month'])['Forecast']
    on_hand, backlog = None, 0.0
    pipeline = {}  # arrival month -> qty
    recs = []
    for i, m in enumerate(MONTHS):
        if m < REVIEWS[0]: continue
        if m in pipeline:
            oh_in = pipeline.pop(m)
        else:
            oh_in = 0.0
        if on_hand is None:
            # warm start: initial stock = first S (no pipeline)
            f5 = fu.loc[m].sum() if m in fu.index.get_level_values(0) else panel[u].mean() * 5
            on_hand = f5 + Z[sl] * sigma_at(u, mod, m)
        on_hand += oh_in
        # review?
        if m in REVIEWS:
            try:
                f5 = fu.loc[m].sum()
            except KeyError:
                f5 = panel.loc[:m - pd.DateOffset(months=1), u].tail(12).mean() * 5
            S = max(0.0, f5 + Z[sl] * sigma_at(u, mod, m))
            IP = on_hand + sum(pipeline.values()) - backlog
            q = max(0.0, S - IP)
            if lots: q = lot_round(q, u)
            if q > 0: pipeline[m + pd.DateOffset(months=L)] = pipeline.get(m + pd.DateOffset(months=L), 0.0) + q
        # demand
        d = panel.loc[m, u]
        avail = on_hand
        served_backlog = min(avail, backlog)
        avail -= served_backlog; backlog -= served_backlog
        served = min(avail, d)
        short = d - served
        on_hand = avail - served
        backlog += short
        recs.append({'Month': m, 'demand': d, 'served': served, 'short': short,
                     'end_oh': on_hand, 'end_bl': backlog})
    df = pd.DataFrame(recs).set_index('Month')
    ev = df.loc[EVAL_START:EVAL_END]
    D = ev.demand.sum()
    fill = ev.served.sum() / D if D > 0 else 1.0
    hold = ev.end_oh.mean()
    bl = ev.end_bl.mean()
    cost = ev.end_oh.sum() * 1.0 + ev.end_bl.sum() * ratio
    return {'Unit': u, 'Model': mod, 'SL': sl, 'ratio': ratio, 'lots': lots,
            'fill_rate': fill, 'avg_oh': hold, 'avg_bl': bl, 'total_cost': cost,
            'demand_2024': D}

results = []
for u, mod, sl in itertools.product(UNITS, MODELS, [0.90, 0.95, 0.98]):
    for ratio in [2, 5, 10, 20, 50]:
        results.append(simulate(u, mod, sl, ratio, lots=True))
    results.append(simulate(u, mod, sl, 10, lots=False))
res = pd.DataFrame(results)
res.to_csv(os.path.join(HERE, 'sim_results.csv'), index=False)
print('saved', res.shape)
