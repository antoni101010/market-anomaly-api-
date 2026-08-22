from __future__ import annotations
import math
import numpy as np
import pandas as pd


def _cols(signals,horizon):
    h=int(horizon)
    r=f'future_{h}d_net_pct' if f'future_{h}d_net_pct' in signals.columns else f'future_{h}d_pct'
    s=f'spy_future_{h}d_net_pct' if f'spy_future_{h}d_net_pct' in signals.columns else f'spy_future_{h}d_pct'
    return r,s

def bootstrap_excess(signals,horizon=60,n_boot=3000,seed=42):
    if signals is None or signals.empty:
        return {'n':0,'mean_excess':0,'ci95_low':0,'ci95_high':0,'prob_gt_zero':0}
    rc,sc=_cols(signals,horizon)
    x=(pd.to_numeric(signals[rc],errors='coerce')-pd.to_numeric(signals[sc],errors='coerce')).dropna().values
    if len(x)==0:
        return {'n':0,'mean_excess':0,'ci95_low':0,'ci95_high':0,'prob_gt_zero':0}
    rng=np.random.default_rng(seed)
    means=np.empty(int(n_boot))
    for i in range(int(n_boot)):
        means[i]=rng.choice(x,size=len(x),replace=True).mean()
    return {
        'n':int(len(x)),'mean_excess':float(x.mean()),
        'ci95_low':float(np.quantile(means,0.025)),'ci95_high':float(np.quantile(means,0.975)),
        'prob_gt_zero':float((means>0).mean()*100)
    }

def sign_flip_pvalue(signals,horizon=60,n_perm=5000,seed=123):
    if signals is None or signals.empty: return 1.0
    rc,sc=_cols(signals,horizon)
    x=(pd.to_numeric(signals[rc],errors='coerce')-pd.to_numeric(signals[sc],errors='coerce')).dropna().values
    if len(x)<2: return 1.0
    obs=float(x.mean())
    rng=np.random.default_rng(seed)
    vals=[]
    for _ in range(int(n_perm)):
        signs=rng.choice([-1.0,1.0],size=len(x))
        vals.append(float((x*signs).mean()))
    vals=np.asarray(vals)
    return float((1+(np.abs(vals)>=abs(obs)).sum())/(len(vals)+1))

def benjamini_hochberg(pvalues):
    p=np.asarray(pvalues,dtype=float)
    n=len(p)
    if n==0:return np.array([])
    order=np.argsort(p); ranked=p[order]
    adj=np.empty(n); prev=1.0
    for i in range(n-1,-1,-1):
        rank=i+1
        val=min(prev, ranked[i]*n/rank)
        adj[i]=val; prev=val
    out=np.empty(n); out[order]=np.clip(adj,0,1)
    return out

def group_analysis(signals,group_col,horizon=60,min_n=5):
    if signals is None or signals.empty or group_col not in signals.columns:return pd.DataFrame()
    rc,sc=_cols(signals,horizon)
    rows=[]
    for g,x in signals.groupby(group_col,dropna=False):
        r=pd.to_numeric(x[rc],errors='coerce'); s=pd.to_numeric(x[sc],errors='coerce'); ex=(r-s).dropna()
        if len(ex)<int(min_n): continue
        p=sign_flip_pvalue(x,horizon,n_perm=1500,seed=abs(hash(str(g)))%10000)
        rows.append({
            group_col:str(g),'signals':int(len(ex)),'avg_return':float(r.loc[ex.index].mean()),
            'avg_spy':float(s.loc[ex.index].mean()),'avg_excess':float(ex.mean()),
            'beat_spy_rate':float((ex>0).mean()*100),'p_value':p
        })
    out=pd.DataFrame(rows)
    if not out.empty:
        out['p_fdr']=benjamini_hochberg(out['p_value'].values)
        out=out.sort_values('avg_excess',ascending=False)
    return out

def cohort_curve(signals,horizon=60):
    if signals is None or signals.empty:return pd.DataFrame(),{}
    rc,sc=_cols(signals,horizon)
    x=signals.copy(); x['signal_date']=pd.to_datetime(x['signal_date'])
    x['ret']=pd.to_numeric(x[rc],errors='coerce'); x['spy']=pd.to_numeric(x[sc],errors='coerce')
    g=x.groupby('signal_date').agg(strategy_return=('ret','mean'),benchmark_return=('spy','mean'),signals=('ticker','count')).dropna().reset_index()
    if g.empty:return g,{}
    g['strategy_equity']=(1+g['strategy_return']/100).cumprod()
    g['benchmark_equity']=(1+g['benchmark_return']/100).cumprod()
    peak=g['strategy_equity'].cummax(); dd=(g['strategy_equity']/peak-1)*100
    g['strategy_drawdown_pct']=dd
    stats={
        'cohorts':int(len(g)),
        'strategy_total_pct':float((g['strategy_equity'].iloc[-1]-1)*100),
        'benchmark_total_pct':float((g['benchmark_equity'].iloc[-1]-1)*100),
        'max_drawdown_pct':float(dd.min()),
    }
    return g,stats

def infer_regime(spy_trailing_60d_pct, spy_above_200d=None):
    try: r=float(spy_trailing_60d_pct)
    except Exception:return 'Unknown'
    if spy_above_200d is not None:
        if bool(spy_above_200d) and r>=5:return 'Bull'
        if (not bool(spy_above_200d)) and r<=-5:return 'Bear'
    if r>=5:return 'Bull'
    if r<=-5:return 'Bear'
    return 'Neutral'

def weight_stability(folds: pd.DataFrame):
    if folds is None or folds.empty:return pd.DataFrame(),0.0
    cols=[c for c in folds.columns if c.startswith('w_')]
    if not cols:return pd.DataFrame(),0.0
    rows=[]
    cvs=[]
    for c in cols:
        v=pd.to_numeric(folds[c],errors='coerce').dropna()
        mean=float(v.mean()) if len(v) else 0
        sd=float(v.std(ddof=1)) if len(v)>1 else 0
        cv=sd/mean if mean>1e-9 else np.nan
        if not np.isnan(cv):cvs.append(cv)
        rows.append({'feature':c[2:],'mean_weight':mean,'std_weight':sd,'cv':cv})
    stability=max(0.0,min(100.0,100*(1-np.nanmean(cvs)))) if cvs else 0.0
    return pd.DataFrame(rows).sort_values('mean_weight',ascending=False),float(stability)
