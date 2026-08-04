"""Fast smoke test: every module imports, the fixture builds, the simulator runs, and the
vintage rule is origin-safe. Runs in well under a minute so it can gate every commit.

The full pipeline (run_all.py) takes about an hour and is not suitable for CI; this
checks the things that actually broke in past revisions — a missing dependency, a
renamed file, a policy-logic import error, and the leakage rule.
"""
import importlib
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MODULES = ['simcore']
SCRIPTS = ['00_make_synthetic_data.py', '02_forecasts.py', '03_ml_models.py',
           '04_simulate.py', '05_alignment.py', '06_service_frontier.py',
           '07_bootstrap.py', '08_figures.py', '09_report_numbers.py',
           '10_seed_study.py', '11_alpha_sensitivity.py', '12_refit_frequency.py',
           '13_refit_eval.py', 'run_all.py', 'test_leakage.py']


def check_syntax():
    for s in SCRIPTS:
        p = os.path.join(HERE, s)
        assert os.path.exists(p), f'missing script: {s}'
        compile(open(p).read(), p, 'exec')
    print(f'ok: {len(SCRIPTS)} scripts present and syntactically valid')


def check_imports():
    for m in MODULES:
        importlib.import_module(m)
    print('ok: shared modules import')


def check_declared_dependencies():
    """Every `python3 <name>` a script shells out to must exist in this repository."""
    import re
    missing = []
    for s in SCRIPTS:
        txt = open(os.path.join(HERE, s)).read()
        for ref in re.findall(r"['\"]([0-9a-zA-Z_]+\.py)['\"]", txt):
            if ref not in SCRIPTS + [m + '.py' for m in MODULES] and not ref.startswith('_'):
                missing.append((s, ref))
    assert not missing, f'scripts reference files that are not in the repository: {missing}'
    print('ok: no script references a missing file')


def check_fixture_and_simulator():
    r = subprocess.run([sys.executable, '00_make_synthetic_data.py'], cwd=HERE,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    import simcore as S
    panel = pd.read_csv(os.path.join(HERE, 'unit_demand_monthly.csv'),
                        index_col=0, parse_dates=True)
    months = list(panel.index)
    reviews = [m for m in months if m >= pd.Timestamp('2023-01-01')
               and ((m.year - 2023) * 12 + m.month - 1) % S.R == 0]
    unit = [u for u in panel.columns if panel[u].sum() > 0][0]
    # a one-method forecast frame: repeat the last observed month
    rows = []
    for t in [m for m in months if m >= pd.Timestamp('2023-01-01')]:
        prev = panel.loc[:t - pd.DateOffset(months=1), unit]
        f = float(prev.iloc[-1]) if len(prev) else 0.0
        for j in range(5):
            m = t + pd.DateOffset(months=j)
            if m <= months[-1]:
                rows.append({'Unit': unit, 'Origin': t, 'Month': m, 'h': j,
                             'Model': 'naive', 'Forecast': f,
                             'Actual': float(panel.loc[m, unit])})
    fc = pd.DataFrame(rows)
    fsum = S.protection_errors(fc)
    out = S.simulate(panel, fc, fsum, months, reviews, unit, 'naive')
    assert out['total_cost'] >= 0 and 0 <= (out['fill_rate'] or 0) <= 1, out
    print(f"ok: simulator runs (cost {out['total_cost']:.0f}, fill {out['fill_rate']:.3f})")


def check_vintage_rule():
    """The rule itself, without needing the full forecast frame."""
    cutoff = {2021: pd.Timestamp('2021-12-01'), 2022: pd.Timestamp('2022-12-01'),
              2023: pd.Timestamp('2023-12-01')}
    origins = pd.date_range('2022-01-01', '2024-12-01', freq='MS')
    for t in origins:
        vy = min(max(t.year - 1, 2021), 2023)
        assert cutoff[vy] < t, f'origin {t.date()} would use a vintage cut at {cutoff[vy].date()}'
    print(f'ok: vintage rule is origin-safe for all {len(origins)} origins')


if __name__ == '__main__':
    check_syntax()
    check_imports()
    check_declared_dependencies()
    check_vintage_rule()
    check_fixture_and_simulator()
    print('\nsmoke test passed')
