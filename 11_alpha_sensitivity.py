"""N-01 / M-03: sensitivity of the intermittent estimators to the smoothing constant."""
import pandas as pd, numpy as np, os, json, importlib.util, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
panel = pd.read_csv(os.path.join(HERE,'unit_demand_monthly.csv'), index_col=0, parse_dates=True)
fc0 = pd.read_parquet(os.path.join(HERE,'forecasts.parquet'))
fc0.to_parquet(os.path.join(HERE,'_fc_alpha_backup.parquet'))  # baseline, restored at the end
MONTHS=list(panel.index); ORIGINS=[m for m in MONTHS if m>=pd.Timestamp('2022-01-01')]
UNITS=[u for u in panel.columns if panel[u].sum()>0]

def variants(y, a):
    y=np.asarray(y,float); nz=np.nonzero(y)[0]
    if len(nz)==0: return {'croston':0.0,'sba':0.0,'tsb':0.0}
    z=y[nz[0]]; p=1.0; first=True; q=1
    for i in range(nz[0]+1,len(y)):
        if y[i]>0:
            z=a*y[i]+(1-a)*z; p=a*q+(1-a)*p if not first else q; first=False; q=1
        else: q+=1
    cro=z/max(p,1.0)
    prob=1.0 if y[0]>0 else 0.0; zt=y[nz[0]]
    for i in range(1,len(y)):
        if y[i]>0: prob=a*1+(1-a)*prob; zt=a*y[i]+(1-a)*zt
        else: prob=(1-a)*prob
    return {'croston':cro,'sba':cro*(1-a/2),'tsb':prob*zt}

act = panel.stack().rename('Actual').reset_index(); act.columns=['Month','Unit','Actual']
out=[]
for a in (0.05,0.10,0.20,0.30):
    rows=[]
    for t in ORIGINS:
        he=t-pd.DateOffset(months=1)
        for u in UNITS:
            v=variants(panel.loc[:he,u].values,a)
            for j in range(5):
                m=t+pd.DateOffset(months=j)
                if m>MONTHS[-1]: continue
                for k,val in v.items(): rows.append((u,t,m,j,k,val))
    new=pd.DataFrame(rows,columns=['Unit','Origin','Month','h','Model','Forecast']).merge(act,on=['Unit','Month'],how='left')
    keep=fc0[~fc0.Model.isin(['croston','sba','tsb'])]
    pd.concat([keep,new],ignore_index=True).to_parquet(os.path.join(HERE,'forecasts.parquet'))
    # R3-M-17: os.system hides failures; delete the grid first and check the return code
    grid = os.path.join(HERE, 'sim_results_audit.csv')
    if os.path.exists(grid):
        os.remove(grid)
    res = subprocess.run(['python3', os.path.join(HERE, '04_simulate.py')], cwd=HERE,
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit(f'alpha {a}: 04_simulate.py failed\n{res.stderr[-3000:]}')
    if not os.path.exists(grid):
        raise SystemExit(f'alpha {a}: 04_simulate.py produced no grid')
    r=pd.read_csv(grid)
    b=r[(r.markup==-1)&(r.SL==0.95)&(r.ratio==10)&r.lots&(r.term_k==0)&(r.eval_end=='2024-11-01')]
    s=b.groupby('Model').total_cost.sum(); idx=100*s/s['mitra']
    for m in ('croston','sba','tsb'): out.append({'alpha':a,'Model':m,'cost_index':round(idx[m],1)})
    print('alpha',a,{m:round(idx[m],1) for m in ('croston','sba','tsb')},flush=True)
pd.DataFrame(out).to_csv(os.path.join(HERE,'alpha_sensitivity.csv'),index=False)
# restore the baseline (alpha = 0.1) forecasts AND the downstream result tables
fc0.to_parquet(os.path.join(HERE, 'forecasts.parquet'))
os.remove(os.path.join(HERE, '_fc_alpha_backup.parquet'))
res = subprocess.run(['python3', os.path.join(HERE, '04_simulate.py')], cwd=HERE,
                     capture_output=True, text=True)
if res.returncode != 0:
    raise SystemExit('restoring the baseline grid failed\n' + res.stderr[-3000:])
print('done')
