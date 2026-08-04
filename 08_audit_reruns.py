"""Round-1 audit reruns.

Adds, relative to simulate.py:
  * M-01  the company's ACTUAL current policy: annual plan + fixed percentage markup
          (10 / 20 / 30 %), i.e. no error calibration.
  * B-03  per-unit achieved cycle service (share of replenishment cycles with no
          shortage) alongside fill rate; bias / RMS / SD of protection-interval errors.
  * B-04  terminal accounting: a holding charge of k months on ending on-hand and
          pipeline (k = 0, 2.5, 5) and an evaluation window restricted to reviews
          whose full protection interval is observable.
  * B-06  all-zero units excluded from every metric aggregation; n_valid reported.
"""
import pandas as pd, numpy as np, os, itertools, json
HERE = os.path.dirname(os.path.abspath(__file__))
panel = pd.read_csv(os.path.join(HERE, 'unit_demand_monthly.csv'), index_col=0, parse_dates=True)
fc = pd.read_parquet(os.path.join(HERE, 'forecasts.parquet'))
mitra = json.load(open(os.path.join(HERE, 'mitra_annual.json')))
MONTHS = list(panel.index)
L, R = 3, 2
Z = {0.90: 1.2816, 0.95: 1.6449, 0.98: 2.0537}
REVIEWS = [m for m in MONTHS if m >= pd.Timestamp('2023-01-01') and ((m.year - 2023) * 12 + m.month - 1) % R == 0]
EVAL_START, EVAL_END = pd.Timestamp('2024-01-01'), pd.Timestamp('2024-12-01')
# reviews whose whole protection interval falls inside the data horizon
LAST_FULL_REVIEW = pd.Timestamp('2024-08-01')
UNITS_ALL = list(panel.columns)
UNITS = [u for u in UNITS_ALL if panel[u].sum() > 0]          # B-06: drop all-zero units
MODELS = sorted(fc.Model.unique())

fsum = (fc.groupby(['Unit', 'Model', 'Origin'])
          .agg(F5=('Forecast', 'sum'), A5=('Actual', 'sum'), n=('Forecast', 'size'))
          .reset_index())
fsum = fsum[fsum.n == 5]
fsum['err'] = fsum.F5 - fsum.A5


def sigma_at(u, mod, t):
    cut = t - pd.DateOffset(months=5)
    e = fsum[(fsum.Unit == u) & (fsum.Model == mod) & (fsum.Origin <= cut)]['err'].tail(12)
    if len(e) >= 6:
        return float(np.sqrt((e ** 2).mean()))
    hist = panel.loc[:t - pd.DateOffset(months=1), u].rolling(5).sum().dropna()
    return float(hist.std(ddof=1)) if len(hist) > 3 else float(panel[u].std())


def lot_round(q, unit):
    if q <= 0:
        return 0.0
    base = 25025.0 if unit.startswith('BR') else 20.0
    return float(np.ceil(q / base) * base)


def simulate(u, mod, sl, ratio, lots=True, markup=None, term_k=0.0, eval_end=EVAL_END):
    """markup=None -> error-calibrated buffer; markup=0.2 -> fixed 20% of forecast."""
    fu = fc[(fc.Unit == u) & (fc.Model == mod)].set_index(['Origin', 'Month'])['Forecast']
    on_hand, backlog, pipeline = None, 0.0, {}
    recs = []
    cycle_flags = []            # per replenishment cycle: 1 if no shortage in the cycle
    cyc_short = False
    for m in MONTHS:
        if m < REVIEWS[0]:
            continue
        on_hand = (on_hand or 0.0) + pipeline.pop(m, 0.0) if on_hand is not None else None
        if on_hand is None:
            try:
                f5 = fu.loc[m].sum()
            except KeyError:
                f5 = panel[u].mean() * 5
            buf = markup * f5 if markup is not None else Z[sl] * sigma_at(u, mod, m)
            on_hand = max(0.0, f5 + buf) + pipeline.pop(m, 0.0)
        if m in REVIEWS:
            if cycle_flags or m != REVIEWS[0]:
                cycle_flags.append(0 if cyc_short else 1)
            cyc_short = False
            try:
                f5 = fu.loc[m].sum()
            except KeyError:
                f5 = panel.loc[:m - pd.DateOffset(months=1), u].tail(12).mean() * 5
            buf = markup * f5 if markup is not None else Z[sl] * sigma_at(u, mod, m)
            S = max(0.0, f5 + buf)
            IP = on_hand + sum(pipeline.values()) - backlog
            q = max(0.0, S - IP)
            if lots:
                q = lot_round(q, u)
            if q > 0:
                arr = m + pd.DateOffset(months=L)
                pipeline[arr] = pipeline.get(arr, 0.0) + q
        d = panel.loc[m, u]
        avail = on_hand
        served_bl = min(avail, backlog)
        avail -= served_bl
        backlog -= served_bl
        served = min(avail, d)
        short = d - served
        if short > 1e-9:
            cyc_short = True
        on_hand = avail - served
        backlog += short
        recs.append({'Month': m, 'demand': d, 'served': served, 'short': short,
                     'end_oh': on_hand, 'end_bl': backlog,
                     'pipe': sum(v for k, v in pipeline.items() if k > m)})
    cycle_flags.append(0 if cyc_short else 1)
    df = pd.DataFrame(recs).set_index('Month')
    ev = df.loc[EVAL_START:eval_end]
    D = ev.demand.sum()
    fill = ev.served.sum() / D if D > 0 else np.nan
    # achieved cycle service over cycles that start inside the evaluation window
    cyc_idx = [i for i, r in enumerate(REVIEWS) if EVAL_START <= r <= eval_end]
    css = np.mean([cycle_flags[i] for i in cyc_idx if i < len(cycle_flags)]) if cyc_idx else np.nan
    terminal = term_k * (ev.end_oh.iloc[-1] + ev['pipe'].iloc[-1]) if len(ev) else 0.0
    cost = ev.end_oh.sum() * 1.0 + ev.end_bl.sum() * ratio + terminal
    return {'Unit': u, 'Model': mod, 'SL': sl, 'ratio': ratio, 'lots': lots,
            'markup': -1 if markup is None else markup, 'term_k': term_k,
            'eval_end': str(eval_end.date()),
            'fill_rate': fill, 'cycle_service': css, 'avg_oh': ev.end_oh.mean(),
            'avg_bl': ev.end_bl.mean(), 'total_cost': cost, 'demand_2024': D,
            'end_oh': ev.end_oh.iloc[-1] if len(ev) else 0.0,
            'end_pipe': ev['pipe'].iloc[-1] if len(ev) else 0.0}


rows = []
# 1. calibrated-buffer methods over the standard grid
for u, mod, sl in itertools.product(UNITS, MODELS, [0.90, 0.95, 0.98]):
    for ratio in [2, 5, 10, 20, 50]:
        rows.append(simulate(u, mod, sl, ratio))
    rows.append(simulate(u, mod, sl, 10, lots=False))
# 2. terminal-accounting and observable-window sensitivities at the base case
for u, mod in itertools.product(UNITS, MODELS):
    for k in [2.5, 5.0]:
        rows.append(simulate(u, mod, 0.95, 10, term_k=k))
    rows.append(simulate(u, mod, 0.95, 10, eval_end=LAST_FULL_REVIEW))
# 3. M-01: the company's actual current policy (annual plan + fixed markup)
for u, mk in itertools.product(UNITS, [0.10, 0.20, 0.30]):
    for ratio in [2, 5, 10, 20, 50]:
        rows.append(simulate(u, 'mitra', 0.95, ratio, markup=mk))
    for k in [2.5, 5.0]:
        rows.append(simulate(u, 'mitra', 0.95, 10, markup=mk, term_k=k))
    rows.append(simulate(u, 'mitra', 0.95, 10, markup=mk, eval_end=LAST_FULL_REVIEW))

res = pd.DataFrame(rows)
res.to_csv(os.path.join(HERE, 'sim_results_audit.csv'), index=False)
print('rows:', len(res), '| units:', len(UNITS), '(excluded all-zero:',
      len(UNITS_ALL) - len(UNITS), ')')

# ---- error decomposition (B-03): bias, RMS, SD of protection-interval errors ----
dec = []
for u in UNITS:
    for mod in MODELS:
        e = fsum[(fsum.Unit == u) & (fsum.Model == mod) &
                 (fsum.Origin >= '2024-01-01') & (fsum.Origin <= '2024-12-01')]['err']
        if len(e) < 3:
            continue
        dec.append({'Unit': u, 'Model': mod, 'bias': e.mean(),
                    'sd': e.std(ddof=1), 'rms': np.sqrt((e ** 2).mean()), 'n': len(e)})
pd.DataFrame(dec).to_csv(os.path.join(HERE, 'error_decomposition.csv'), index=False)
print('error decomposition written')
