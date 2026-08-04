"""B-07: variability of the ML results across random seeds."""
import subprocess, re, os, shutil, pandas as pd, numpy as np, json
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE,'03_ml_models.py')).read()
out = []
for seed in (42, 7, 2026):
    s = src.replace('random_state=42', f'random_state={seed}')
    tmp = os.path.join(HERE, f'_ml_seed{seed}.py')
    open(tmp,'w').write(s)
    shutil.copy(os.path.join(HERE,'forecasts.parquet'), os.path.join(HERE,'_fc_backup.parquet'))
    # R3-M-17: fail loudly, and never read an output that a failed run left behind
    grid = os.path.join(HERE, 'sim_results_audit.csv')
    if os.path.exists(grid):
        os.remove(grid)
    for cmd in ([tmp], [os.path.join(HERE, '04_simulate.py')]):
        res = subprocess.run(['python3'] + cmd, cwd=HERE, capture_output=True, text=True)
        if res.returncode != 0:
            raise SystemExit(f'seed {seed}: {cmd[0]} failed\n{res.stderr[-3000:]}')
    if not os.path.exists(grid):
        raise SystemExit(f'seed {seed}: 04_simulate.py produced no grid')
    r = pd.read_csv(grid)
    b = r[(r.markup==-1)&(r.SL==0.95)&(r.ratio==10)&r.lots&(r.term_k==0)&(r.eval_end=='2024-11-01')]
    s2 = b.groupby('Model').total_cost.sum(); idx = 100*s2/s2['mitra']
    for m in ('rf','xgb','hybrid'):
        out.append({'seed':seed,'Model':m,'cost_index':round(idx[m],1)})
    print('seed', seed, {m: round(idx[m],1) for m in ('rf','xgb','hybrid')}, flush=True)
    shutil.copy(os.path.join(HERE,'_fc_backup.parquet'), os.path.join(HERE,'forecasts.parquet'))
    os.remove(tmp)
pd.DataFrame(out).to_csv(os.path.join(HERE,'seed_variability.csv'), index=False)
# the loop restores forecasts.parquet after each seed; rerun the audit once more so that
# sim_results_audit.csv and everything derived from it describe the baseline seed again
os.remove(os.path.join(HERE, '_fc_backup.parquet'))
res = subprocess.run(['python3', os.path.join(HERE, '04_simulate.py')], cwd=HERE,
                     capture_output=True, text=True)
if res.returncode != 0:
    raise SystemExit('restoring the baseline grid failed\n' + res.stderr[-3000:])
print('done')
