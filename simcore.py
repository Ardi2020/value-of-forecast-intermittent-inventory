"""Single implementation of the (R,S) simulator, shared by every downstream script.

Keeping one copy is deliberate: the Round-2 audit found that separate copies of the
policy logic had drifted, so audit_rerun, the service frontier, the bootstrap and the
refit study all import from here.

Policy: periodic review every R = 2 months, lead time L = 3 months, protection interval
L + R = 5 months, order-up-to level S = sum of the five-month forecast + buffer, order
quantity rounded up to the shipping lot, unmet demand backordered.

Two windows, deliberately separate (R3-B-15):

  DECISION window  reviews up to and including LAST_DECISION = July 2024. July is the
                   last review whose whole five-month protection interval is observed in
                   data ending December 2024, so every ordering decision evaluated here
                   has a fully observed consequence. Later reviews are not executed in
                   the primary configuration; `last_decision=None` re-enables them.

  OUTCOME window   January to November 2024, the months over which cost and service are
                   accumulated. November is where the July decision's protection interval
                   closes.

The two are reported separately because they answer different questions: the decision
window defines which choices are being judged, the outcome window defines over what
period their consequences are counted. Ending on-hand and pipeline at the close of the
outcome window are reported explicitly and charged in the terminal sensitivity.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

L, R = 3, 2
Z = {0.90: 1.2816, 0.95: 1.6449, 0.98: 2.0537}
EVAL_START = pd.Timestamp('2024-01-01')
EVAL_END = pd.Timestamp('2024-11-01')            # primary outcome window
EVAL_END_CALENDAR = pd.Timestamp('2024-12-01')   # sensitivity
LAST_DECISION = pd.Timestamp('2024-07-01')       # last fully observable review
LAST_FULL_REVIEW = LAST_DECISION                 # backwards-compatible alias


def load(fc_name='forecasts.parquet'):
    panel = pd.read_csv(os.path.join(HERE, 'unit_demand_monthly.csv'),
                        index_col=0, parse_dates=True)
    fc = pd.read_parquet(os.path.join(HERE, fc_name))
    months = list(panel.index)
    reviews = [m for m in months if m >= pd.Timestamp('2023-01-01')
               and ((m.year - 2023) * 12 + m.month - 1) % R == 0]
    units = [u for u in panel.columns if panel[u].sum() > 0]   # B-06
    return panel, fc, months, reviews, units


def protection_errors(fc):
    """Realised five-month-sum forecast errors, one row per (unit, model, origin)."""
    fs = (fc.groupby(['Unit', 'Model', 'Origin'])
            .agg(F5=('Forecast', 'sum'), A5=('Actual', 'sum'), n=('Forecast', 'size'))
            .reset_index())
    fs = fs[fs.n == 5].copy()
    fs['err'] = fs.F5 - fs.A5
    return fs


def sigma_at(fsum, panel, u, mod, t):
    """RMS of the last twelve error windows completed strictly before t."""
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


def simulate(panel, fc, fsum, months, reviews, u, mod, *, z=None, sl=0.95, ratio=10,
             lots=True, markup=None, term_k=0.0, eval_end=EVAL_END,
             last_decision=LAST_DECISION):
    """One unit under one method.

    z              explicit buffer multiplier; overrides sl when given (service frontier)
    markup         fixed fraction of the five-month forecast instead of an error-calibrated
                   buffer, i.e. the company's current rule
    term_k         holding charge, in months, on ending on-hand plus pipeline
    last_decision  no order is placed at a review after this month; None runs every review
    """
    zval = Z[sl] if z is None else z
    fu = fc[(fc.Unit == u) & (fc.Model == mod)].set_index(['Origin', 'Month'])['Forecast']
    on_hand, backlog, pipeline = None, 0.0, {}
    recs, cycle_flags, cyc_short = [], [], False
    for m in months:
        if m < reviews[0]:
            continue
        on_hand = (on_hand or 0.0) + pipeline.pop(m, 0.0) if on_hand is not None else None
        if on_hand is None:
            try:
                f5 = fu.loc[m].sum()
            except KeyError:
                f5 = panel[u].mean() * 5
            buf = markup * f5 if markup is not None else zval * sigma_at(fsum, panel, u, mod, m)
            on_hand = max(0.0, f5 + buf) + pipeline.pop(m, 0.0)
        if m in reviews and (last_decision is None or m <= last_decision):
            if cycle_flags or m != reviews[0]:
                cycle_flags.append(0 if cyc_short else 1)
            cyc_short = False
            try:
                f5 = fu.loc[m].sum()
            except KeyError:
                f5 = panel.loc[:m - pd.DateOffset(months=1), u].tail(12).mean() * 5
            buf = markup * f5 if markup is not None else zval * sigma_at(fsum, panel, u, mod, m)
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
    cyc_idx = [i for i, r in enumerate(reviews) if EVAL_START <= r <= eval_end]
    css = np.mean([cycle_flags[i] for i in cyc_idx if i < len(cycle_flags)]) if cyc_idx else np.nan
    terminal = term_k * (ev.end_oh.iloc[-1] + ev['pipe'].iloc[-1]) if len(ev) else 0.0
    cost = ev.end_oh.sum() * 1.0 + ev.end_bl.sum() * ratio + terminal
    return {'Unit': u, 'Model': mod, 'SL': sl, 'z': zval, 'ratio': ratio, 'lots': lots,
            'markup': -1 if markup is None else markup, 'term_k': term_k,
            'eval_end': str(eval_end.date()),
            'last_decision': 'none' if last_decision is None else str(last_decision.date()),
            'fill_rate': fill, 'cycle_service': css,
            'avg_oh': ev.end_oh.mean(), 'avg_bl': ev.end_bl.mean(), 'total_cost': cost,
            'demand_2024': D, 'end_oh': ev.end_oh.iloc[-1] if len(ev) else 0.0,
            'end_pipe': ev['pipe'].iloc[-1] if len(ev) else 0.0}


NAME = {'mitra': 'Partner annual', 'naive': 'Naive', 'ses': 'SES', 'croston': 'Croston',
        'sba': 'SBA', 'tsb': 'TSB', 'arima': 'ARIMA', 'rf': 'Random Forest',
        'xgb': 'XGBoost (Tweedie)', 'hybrid': 'Hybrid cls-reg'}
