import pandas as pd
from optimizer import find_best_weights
from backtest import select_signals,summarize_signals

def walk_forward_validate(dataset,horizon=60,train_years=3,test_months=6,iterations=80,threshold=50,top_n=3,cooldown=2):
    if dataset is None or dataset.empty: raise ValueError("Dataset vuoto.")
    x=dataset.copy(); x["signal_date"]=pd.to_datetime(x["signal_date"]); start=x["signal_date"].min(); end=x["signal_date"].max()
    test_start=start+pd.DateOffset(years=int(train_years)); folds=[]; all_sig=[]; fold_no=0
    while test_start<=end:
        test_end=min(test_start+pd.DateOffset(months=int(test_months)),end+pd.Timedelta(days=1))
        train=x[(x.signal_date<test_start)&(x.signal_date>=test_start-pd.DateOffset(years=int(train_years)))].copy()
        test=x[(x.signal_date>=test_start)&(x.signal_date<test_end)].copy()
        if train.signal_date.nunique()>=10 and test.signal_date.nunique()>=2:
            w,_=find_best_weights(train,horizon,iterations,threshold,top_n,cooldown,seed=42+fold_no)
            sig=select_signals(test,w,threshold,top_n,cooldown); m=summarize_signals(sig,horizon)
            folds.append({"fold":fold_no+1,"train_start":train.signal_date.min(),"train_end":train.signal_date.max(),"test_start":test_start,"test_end":test_end-pd.Timedelta(days=1),**m,**{f"w_{k}":v for k,v in w.items()}})
            if not sig.empty:
                sig=sig.copy(); sig["fold"]=fold_no+1; all_sig.append(sig)
            fold_no+=1
        test_start=test_end
    fold_df=pd.DataFrame(folds); sig_df=pd.concat(all_sig,ignore_index=True) if all_sig else pd.DataFrame()
    overall=summarize_signals(sig_df,horizon) if not sig_df.empty else summarize_signals(pd.DataFrame(),horizon)
    positive_folds=float((fold_df["avg_excess"]>0).mean()*100) if not fold_df.empty else 0.0
    grade="D"
    if overall["signals"]>=80 and overall["ci95_low"]>0 and positive_folds>=70: grade="A"
    elif overall["signals"]>=50 and overall["avg_excess"]>0 and positive_folds>=60: grade="B"
    elif overall["signals"]>=25 and overall["avg_excess"]>0: grade="C"
    return {"folds":fold_df,"signals":sig_df,"overall":overall,"positive_folds_pct":positive_folds,"reliability_grade":grade}
