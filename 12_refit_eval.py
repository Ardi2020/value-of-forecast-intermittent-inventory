"""Evaluate the two refit schedules (yearly vintage vs refit at every origin)
on the same base-case policy, using the audit simulator.

Outputs refit_frequency.csv with cost index (mitra = 100), fill rate, cycle
service and accuracy for rf / xgb / hybrid under both schedules.
"""
import pandas as pd, numpy as np, os, itertools
HERE = os.path.dirname(os.path.abspath(__file__))
panel = pd.read_csv(os.path.join(HERE, 'unit_demand_monthly.csv'), index_col=0, parse_dates=True)
MONTHS = list(panel.index)
L, R = 3, 2
Z = {0.90: 1.2816, 0.95: 1.6449, 0.98: 2.0537}
REVIEWS = [m for m in MONTHS if m >= pd.Timestamp('2023-01-01') and ((m.year - 2023) * 12 + m.month - 1) % R == 0]
EVAL_START, EVAL_END = pd.Timestamp('2024-01-01'), pd.Timestamp('2024-12-01')
UNITS = [u for u in panel.columns if panel[u].sum() > 0]

base_fc = pd.read_parquet(os.path.join(HERE, 'forecasts.parquet'))
refit_ml = pd.read_parquet(os.path.join(HERE, 'forecasts_refit.parquet'))
FRAMES = {'yearly': base_fc,
          'every_origin': pd.concat([base_fc[~base_fc.Model.isin(['rf', 'xgb', 'hybrid'])],
                                     refit_ml], ignore_index=True)}


def prep(fc):
    fs = (fc.groupby(['Unit', 'Model', 'Origin'])
            .agg(F5=('Forecast', 'sum'), A5=('Actual', 'sum'), n=('Forecast', 'size'))
            .reset_index())
    fs = fs[fs.n == 5]
    fs['err'] = fs.F5 - fs.A5
    return fs


def sigma_at(fsum, u, mod, t):
    cut = t - pd.DateOffset(months=5)
    e = fsum[(fsum.Unit == u) & (fsum.Model == mod) & (fsum.Origin <= cut)]['err'].tail(12)
    if len(e) >= 6:
        return float(np.sqrt((e ** 2).mean()))
    hist = panel.loc[:t - pd.DateOffset(months=1), u].rolling(5).sum().dropna()
    return float(hist.std(ddof=1)) if len(hist) > 3 else float(panel[u].std())


def lot_round(q, unit):
    if q <= 0:
        return 0.0
    b = 25025.0 if unit.startswith('BR') else 20.0
    return float(np.ceil(q / b) * b)


def simulate(fc, fsum, u, mod, sl=0.95, ratio=10, term_k=0.0):
    fu = fc[(fc.Unit == u) & (fc.Model == mod)].set_index(['Origin', 'Month'])['Forecast']
    on_hand, backlog, pipeline = None, 0.0, {}
    recs, cycle_flags, cyc_short = [], [], False
    for m in MONTHS:
        if m < REVIEWS[0]:
            continue
        on_hand = (on_hand or 0.0) + pipeline.pop(m, 0.0) if on_hand is not None else None
        if on_hand is None:
            try:
                f5 = fu.loc[m].sum()
            except KeyError:
                f5 = panel[u].mean() * 5
            on_hand = max(0.0, f5 + Z[sl] * sigma_at(fsum, u, mod, m)) + pipeline.pop(m, 0.0)
        if m in REVIEWS:
            if cycle_flags or m != REVIEWS[0]:
                cycle_flags.append(0 if cyc_short else 1)
            cyc_short = False
            try:
                f5 = fu.loc[m].sum()
            except KeyError:
                f5 = panel.loc[:m - pd.DateOffset(months=1), u].tail(12).mean() * 5
            S = max(0.0, f5 + Z[sl] * sigma_at(fsum, u, mod, m))
            q = max(0.0, S - (on_hand + sum(pipeline.values()) - backlog))
            q = lot_round(q, u)
            if q > 0:
                arr = m + pd.DateOffset(months=L)
                pipeline[arr] = pipeline.get(arr, 0.0) + q
        d = panel.loc[m, u]
        avail = on_hand
        sb = min(avail, backlog); avail -= sb; backlog -= sb
        served = min(avail, d); short = d - served
        if short > 1e-9:
            cyc_short = True
        on_hand = avail - served; backlog += short
        recs.append({'Month': m, 'demand': d, 'served': served, 'end_oh': on_hand,
                     'end_bl': backlog, 'pipe': sum(v for k, v in pipeline.items() if k > m)})
    cycle_flags.append(0 if cyc_short else 1)
    ev = pd.DataFrame(recs).set_index('Month').loc[EVAL_START:EVAL_END]
    D = ev.demand.sum()
    cyc_idx = [i for i, r in enumerate(REVIEWS) if EVAL_START <= r <= EVAL_END]
    css = np.mean([cycle_flags[i] for i in cyc_idx if i < len(cycle_flags)]) if cyc_idx else np.nan
    terminal = term_k * (ev.end_oh.iloc[-1] + ev['pipe'].iloc[-1])
    return {'Unit': u, 'Model': mod, 'demand_2024': D,
            'fill_rate': ev.served.sum() / D if D > 0 else np.nan,
            'cycle_service': css, 'total_cost': ev.end_oh.sum() + ev.end_bl.sum() * ratio + terminal}


scales = {}
for u in panel.columns:
    y = panel.loc[:'2023-12-01', u].values
    d = np.abs(np.diff(y))
    scales[u] = {'mae': max(d.mean(), 1e-9), 'rmse': max(np.sqrt((np.diff(y) ** 2).mean()), 1e-9)}


def accuracy(fc, models):
    ev = fc[(fc.Month >= '2024-01-01') & (fc.Month <= '2024-12-01')]
    out = []
    for (u, mod), g in ev.groupby(['Unit', 'Model']):
        if mod not in models or u not in UNITS:
            continue
        e = g.Forecast - g.Actual
        g5 = g.groupby('Origin').agg(F5=('Forecast', 'sum'), A5=('Actual', 'sum'), n=('Forecast', 'size'))
        g5 = g5[g5.n == 5]
        apes = [100.0 if (a < .1 and f > .1) else (0.0 if a < .1 else abs((a - f) / a) * 100)
                for a, f in zip(g.Actual, g.Forecast)]
        out.append({'Unit': u, 'Model': mod,
                    'MASE': np.abs(e).mean() / scales[u]['mae'],
                    'RMSSE': np.sqrt((e ** 2).mean()) / scales[u]['rmse'],
                    'MASE5': (np.abs(g5.F5 - g5.A5).mean() / (scales[u]['mae'] * 5)) if len(g5) else np.nan,
                    'customMAPE': np.mean(apes)})
    return pd.DataFrame(out)


ML = ['rf', 'xgb', 'hybrid']
rows = []
for sched, fc in FRAMES.items():
    fsum = prep(fc)
    sims = pd.DataFrame([simulate(fc, fsum, u, mod, term_k=k)
                         for u, mod, k in itertools.product(UNITS, ML + ['mitra'], [0.0, 2.5])
                         if not (sched == 'yearly' and False)]
                        + [])
    sims['term_k'] = [k for u, mod, k in itertools.product(UNITS, ML + ['mitra'], [0.0, 2.5])]
    acc = accuracy(fc, ML)
    for k in (0.0, 2.5):
        s = sims[sims.term_k == k]
        tot = s.groupby('Model').total_cost.sum()
        piv = s.pivot(index='Unit', columns='Model', values='total_cost')
        rel = piv.div(piv['mitra'], axis=0)
        for mod in ML:
            g = s[s.Model == mod]
            rows.append({'schedule': sched, 'term_k': k, 'Model': mod,
                         'cost_index': round(100 * tot[mod] / tot['mitra'], 1),
                         'cost_index_median_unit': round(100 * rel[mod].median(), 1),
                         'agg_fill': round((g.fill_rate * g.demand_2024).sum() / g.demand_2024.sum(), 3),
                         'fill_median': round(g.fill_rate.median(), 3),
                         'cycle_service_median': round(g.cycle_service.median(), 3),
                         'MASE_median': round(acc[acc.Model == mod].MASE.median(), 3),
                         'MASE5_median': round(acc[acc.Model == mod].MASE5.median(), 3),
                         'customMAPE_median': round(acc[acc.Model == mod].customMAPE.median(), 1)})

out = pd.DataFrame(rows)
out.to_csv(os.path.join(HERE, 'refit_frequency.csv'), index=False)
print(out.to_string(index=False))
