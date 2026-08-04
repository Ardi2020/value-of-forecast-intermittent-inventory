"""Single fail-fast entry point: fixture -> forecasts -> simulation -> tables and figures.

    python run_all.py            # full pipeline on the synthetic fixture
    python run_all.py --quick    # skip the three slow sensitivity studies

Every step must exit zero and must produce the outputs it declares, or the run stops
there with a non-zero exit code. Generated outputs are deleted before the run so that a
stale file can never be mistaken for a fresh one, which is the failure mode the Round-2
audit of this repository identified.

The pipeline runs on the synthetic fixture produced by 00_make_synthetic_data.py. It
demonstrates that every reported analysis is executable end to end; it does not
reproduce the confidential case numbers. The aggregate result tables computed from the
real data are shipped separately under results/.
"""
import argparse
import hashlib
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ('00_make_synthetic_data.py', ['unit_demand_monthly.csv', 'mitra_annual.json', 'exog_monthly.csv'], False),
    ('02_forecasts.py',           ['forecasts.parquet'], False),
    ('03_ml_models.py',           ['forecasts.parquet'], False),
    ('04_simulate.py',            ['sim_results_audit.csv', 'error_decomposition.csv'], False),
    ('05_alignment.py',           ['accuracy.csv', 'intermittency_profile.csv', 'alignment_audit.csv'], False),
    ('06_service_frontier.py',    ['service_frontier.csv', 'matched_service.csv'], False),
    ('07_bootstrap.py',           ['bootstrap_intervals.csv'], False),
    ('08_figures.py',             ['figures/fig3_cost_index.png', 'figures/fig4_service_frontier.png'], False),
    ('09_report_numbers.py',      [], False),
    ('10_seed_study.py',          ['seed_variability.csv'], True),
    ('11_alpha_sensitivity.py',   ['alpha_sensitivity.csv'], True),
    ('12_refit_frequency.py',     ['forecasts_refit.parquet'], True),
    ('13_refit_eval.py',          ['refit_frequency.csv'], True),
]

GENERATED = sorted({o for _, outs, _ in STEPS for o in outs} |
                   {'unit_demand_monthly.csv', 'mitra_annual.json', 'exog_monthly.csv',
                    'forecasts.parquet', 'forecasts_refit.parquet', 'accuracy_calendar.csv'})


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true',
                    help='skip the seed, smoothing-constant and refit studies')
    args = ap.parse_args()

    removed = 0
    for rel in GENERATED:
        p = os.path.join(HERE, rel)
        if os.path.exists(p):
            os.remove(p)
            removed += 1
    print(f'cleared {removed} previously generated file(s)\n', flush=True)

    t0 = time.time()
    for script, outputs, slow in STEPS:
        if slow and args.quick:
            print(f'-- skipped {script} (--quick)')
            continue
        print(f'== {script}', flush=True)
        r = subprocess.run([sys.executable, script], cwd=HERE)
        if r.returncode != 0:
            sys.exit(f'FAILED: {script} exited {r.returncode}')
        missing = [o for o in outputs if not os.path.exists(os.path.join(HERE, o))]
        if missing:
            sys.exit(f'FAILED: {script} did not produce {missing}')
        print(f'   ok ({time.time() - t0:.0f}s elapsed)\n', flush=True)

    print('=== checksums of generated outputs ===')
    for rel in GENERATED:
        p = os.path.join(HERE, rel)
        if os.path.exists(p):
            print(f'  {sha256(p)}  {rel}')
    print(f'\npipeline completed in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
