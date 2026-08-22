import numpy as np
import pandas as pd
from model import WEIGHT_KEYS,normalize_weights,BACKTEST_WEIGHTS
from backtest import select_signals,summarize_signals

def _objective(m):
    if m["signals"]<12:return -999+m["signals"]
    stability=0.04*(m["beat_spy_rate"]-50)
    uncertainty=0.15*max(0,m["avg_excess"]-m["ci95_low"])
    return m["avg_excess"]+stability-uncertainty

def find_best_weights(train,horizon=60,iterations=120,threshold=50,top_n_per_scan=3,cooldown_scans=2,seed=42):
    rng=np.random.default_rng(seed); candidates=[normalize_weights(BACKTEST_WEIGHTS)]
    for _ in range(int(iterations)):
        v=rng.dirichlet(np.ones(len(WEIGHT_KEYS))*1.5); candidates.append({k:float(x) for k,x in zip(WEIGHT_KEYS,v)})
    best=None; best_obj=-1e18; records=[]
    for w in candidates:
        s=select_signals(train,w,threshold,top_n_per_scan,cooldown_scans); m=summarize_signals(s,horizon)
        o=_objective(m); records.append({"objective":o,**w,**{f"train_{k}":v for k,v in m.items()}})
        if o>best_obj: best_obj=o; best=w
    return best,pd.DataFrame(records).sort_values("objective",ascending=False).reset_index(drop=True)

def optimize_weights(dataset,horizon=60,iterations=150,threshold=50,top_n_per_scan=3,cooldown_scans=2,train_ratio=0.70,seed=42):
    dates=sorted(pd.to_datetime(dataset["signal_date"]).dropna().unique())
    if len(dates)<8: raise ValueError("Servono più date.")
    cut=max(1,min(len(dates)-1,int(len(dates)*train_ratio))); split=pd.Timestamp(dates[cut])
    train=dataset[pd.to_datetime(dataset["signal_date"])<split].copy(); test=dataset[pd.to_datetime(dataset["signal_date"])>=split].copy()
    best,leader=find_best_weights(train,horizon,iterations,threshold,top_n_per_scan,cooldown_scans,seed)
    ts=select_signals(train,best,threshold,top_n_per_scan,cooldown_scans); hs=select_signals(test,best,threshold,top_n_per_scan,cooldown_scans)
    return {"weights":best,"split_date":split,"train_metrics":summarize_signals(ts,horizon),"test_metrics":summarize_signals(hs,horizon),"train_signals":ts,"test_signals":hs,"leaderboard":leader}
