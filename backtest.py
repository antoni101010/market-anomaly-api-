
from __future__ import annotations
import pandas as pd
import numpy as np

from indicators import (
    rsi14, drawdown_52w_pct, volume_ratio_20d,
    return_pct, volatility_20d_pct, worst_day_20d_pct
)
from model import technical_components, backtest_score, BACKTEST_WEIGHTS, clamp
from fundamentals import quality_from_metrics
from universe_manager import normalize_universe, active_snapshot

HORIZONS=(20,60,90)

def _base_technical(prices):
    return {
        "drawdown_52w_pct":drawdown_52w_pct(prices),
        "rsi14":rsi14(prices["close"]),
        "volume_ratio_20d":volume_ratio_20d(prices),
        "return_20d_pct":return_pct(prices["close"],20),
        "return_60d_pct":return_pct(prices["close"],60),
        "volatility_20d_pct":volatility_20d_pct(prices["close"]),
        "worst_day_20d_pct":worst_day_20d_pct(prices["close"]),
        "last_close":float(prices["close"].iloc[-1]),
    }

def _prep(df):
    x=df.copy()
    x["datetime"]=pd.to_datetime(x["datetime"],errors="coerce")
    return x.dropna(subset=["datetime","close"]).sort_values("datetime").reset_index(drop=True)

def _idx(df,date):
    if df is None or df.empty:
        return None
    arr=df["datetime"].values
    pos=np.searchsorted(arr,np.datetime64(pd.Timestamp(date)),side="right")-1
    return int(pos) if pos>=0 else None

def _future_return(df,idx,h):
    if idx is None:
        return None
    j=idx+int(h)
    if j>=len(df):
        return None
    p0=float(df.iloc[idx]["close"])
    p1=float(df.iloc[j]["close"])
    return None if p0==0 else (p1/p0-1)*100

def _above_sma(df, idx, periods=200):
    if idx is None or idx < periods-1: return None
    window=pd.to_numeric(df.iloc[idx-periods+1:idx+1]["close"],errors="coerce").dropna()
    if len(window)<periods: return None
    return bool(float(df.iloc[idx]["close"]) >= float(window.mean()))

def _future_return_lifecycle(df, idx, h, signal_date, active_to=None, delisting_return_pct=None):
    """
    If a security delists before the target horizon and a terminal delisting return
    is supplied, combine the pre-delisting price move with that terminal event.
    """
    ordinary = _future_return(df,idx,h)
    target_date = pd.Timestamp(signal_date) + pd.offsets.BDay(int(h))
    end = pd.to_datetime(active_to,errors="coerce")

    if pd.isna(end) or end > target_date:
        return ordinary, False, False

    terminal = pd.to_numeric(pd.Series([delisting_return_pct]),errors="coerce").iloc[0]
    if pd.isna(terminal):
        return None, True, False

    p0=float(df.iloc[idx]["close"])
    # Last bar on/before delisting date.
    j=_idx(df,end)
    if j is None or j < idx or p0 == 0:
        return None, True, False

    pre=float(df.iloc[j]["close"])/p0
    total=(pre*(1+float(terminal)/100.0)-1)*100
    return float(total), True, True

def _active_on(row,date):
    d=pd.Timestamp(date).normalize()
    start=pd.to_datetime(row.get("active_from"),errors="coerce")
    end=pd.to_datetime(row.get("active_to"),errors="coerce")
    if not pd.isna(start) and d<start.normalize(): return False
    if not pd.isna(end) and d>end.normalize(): return False
    return True

def fetch_histories(universe,provider,years=5,adjust="all"):
    u=normalize_universe(universe)
    symbols=list(u["ticker"].astype(str).str.upper())
    sectors=list(u["sector_etf"].astype(str).str.upper().unique())
    needed=sorted(set(symbols+sectors+["SPY"]))
    outputsize=min(5000,max(700,int(years*252+380)))

    histories={}
    if hasattr(provider,"batch_daily_history"):
        try:
            histories=provider.batch_daily_history(needed,outputsize=outputsize,adjust=adjust)
        except TypeError:
            histories=provider.batch_daily_history(needed,outputsize=outputsize)
        except Exception:
            histories={}

    errors={}
    for sym in needed:
        if sym not in histories:
            try:
                try:
                    histories[sym]=provider.daily_history(sym,outputsize=outputsize,adjust=adjust)
                except TypeError:
                    histories[sym]=provider.daily_history(sym,outputsize=outputsize)
            except Exception as e:
                errors[sym]=str(e)
                continue
        try:
            histories[sym]=_prep(histories[sym])
        except Exception as e:
            errors[sym]=str(e)
            histories.pop(sym,None)

    return histories, errors

def build_feature_dataset(
    universe,provider,years=5,scan_every=10,warmup=252,
    fundamental_provider=None,adjust="all",commission_bps=5.0,slippage_bps=10.0
):
    """
    Lifecycle-aware point-in-time dataset.
    The active universe changes by date using active_from/active_to.
    Delisting terminal returns are used when supplied.
    """
    u=normalize_universe(universe)
    histories, history_errors=fetch_histories(u,provider,years,adjust=adjust)
    if "SPY" not in histories:
        raise ValueError("Storico SPY non disponibile.")

    spy=histories["SPY"]
    max_h=max(HORIZONS)
    start=max(int(warmup),len(spy)-int(years*252))
    stop=len(spy)-max_h-1
    if stop<=start:
        raise ValueError("Storico insufficiente.")

    by_ticker={r["ticker"]:r for _,r in u.iterrows()}
    rows=[]
    total_cost_pct=2*(float(commission_bps)+float(slippage_bps))/100.0

    for spy_i in range(start,stop+1,int(scan_every)):
        date=spy.iloc[spy_i]["datetime"]
        spy_tech=_base_technical(spy.iloc[:spy_i+1])
        spy60=spy_tech["return_60d_pct"]
        spy_above_200d=_above_sma(spy,spy_i,200)
        market_regime = "Bull" if (spy60>=5 and spy_above_200d is not False) else ("Bear" if (spy60<=-5 and spy_above_200d is not True) else "Neutral")
        spy_future={h:_future_return(spy,spy_i,h) for h in HORIZONS}

        active = active_snapshot(u,date)
        active_tickers = set(active["ticker"])
        active_count = len(active_tickers)
        with_history = sum(1 for tic in active_tickers if tic in histories)
        history_coverage = (with_history/active_count*100.0) if active_count else 0.0

        sector60={}
        for sector in set(active["sector_etf"].astype(str).str.upper()):
            sh=histories.get(sector)
            si=_idx(sh,date) if sh is not None else None
            sector60[sector]=spy60 if si is None or si<65 else _base_technical(sh.iloc[:si+1])["return_60d_pct"]

        for ticker in active_tickers:
            rowu=by_ticker[ticker]
            hist=histories.get(ticker)
            i=_idx(hist,date) if hist is not None else None
            if i is None or i<warmup:
                continue

            t=_base_technical(hist.iloc[:i+1])
            sector=str(rowu.get("sector_etf","SPY")).upper()
            comps=technical_components(t,spy60,sector60.get(sector,spy60))

            fm={}
            pit_available=False
            pit_filed=None
            if fundamental_provider is not None:
                try:
                    fm=fundamental_provider.metrics_as_of(ticker,date,t["last_close"]) or {}
                    pit_available=bool(fm.get("pit_fact_count",0))
                    pit_filed=fm.get("pit_last_filed_date")
                except Exception:
                    fm={}
            pit_quality=quality_from_metrics(fm) if pit_available else 50.0
            score_quality=clamp(pit_quality)

            future={}
            delist_crossed={}
            delist_used={}
            for h in HORIZONS:
                fr,crossed,used = _future_return_lifecycle(
                    hist,i,h,date,rowu.get("active_to"),rowu.get("delisting_return_pct")
                )
                future[h]=fr
                delist_crossed[h]=crossed
                delist_used[h]=used

            # Keep the record if at least one evaluation horizon is valid.
            if not any(v is not None for v in future.values()):
                continue

            rec={
                "signal_date":date,
                "ticker":ticker,
                "company":rowu.get("company",ticker),
                "sector_etf":sector,
                "price":float(hist.iloc[i]["close"]),
                **t,**comps,
                "pit_quality_score":round(score_quality,2),
                "score_quality_pit":round(score_quality,2),
                "pit_data_available":pit_available,
                "pit_last_filed_date":pit_filed,
                "price_adjustment":adjust,
                "round_trip_cost_pct":round(total_cost_pct,4),
                "universe_source":rowu.get("universe_source","unknown"),
                "active_from":rowu.get("active_from"),
                "active_to":rowu.get("active_to"),
                "delisting_return_pct":rowu.get("delisting_return_pct"),
                "active_universe_size":active_count,
                "history_coverage_pct":round(history_coverage,2),
                "history_missing_symbols":len(history_errors),
                "spy_trailing_60d_pct":round(float(spy60),3),
                "spy_above_200d":spy_above_200d,
                "market_regime":market_regime,
            }

            for k,v in fm.items():
                rec[f"pit_{k}"]=v

            for h in HORIZONS:
                gross=future[h]
                spy_g=spy_future[h]
                rec[f"delist_crossed_{h}d"]=delist_crossed[h]
                rec[f"delist_return_used_{h}d"]=delist_used[h]
                rec[f"future_{h}d_pct"]=gross
                rec[f"spy_future_{h}d_pct"]=spy_g
                if gross is None or spy_g is None:
                    rec[f"future_{h}d_net_pct"]=np.nan
                    rec[f"spy_future_{h}d_net_pct"]=np.nan
                    rec[f"excess_{h}d_pct"]=np.nan
                else:
                    rec[f"future_{h}d_net_pct"]=gross-total_cost_pct
                    rec[f"spy_future_{h}d_net_pct"]=spy_g-total_cost_pct
                    rec[f"excess_{h}d_pct"]=gross-spy_g
            rows.append(rec)

    return pd.DataFrame(rows)

def select_signals(dataset,weights=None,threshold=50,top_n_per_scan=3,cooldown_scans=2):
    if dataset is None or dataset.empty:
        return dataset.copy()
    x=dataset.copy()
    x["backtest_score"]=x.apply(lambda r:backtest_score(r,weights or BACKTEST_WEIGHTS),axis=1)
    x=x[x["backtest_score"]>=float(threshold)].copy()
    if x.empty:
        return x
    x=x.sort_values(["signal_date","backtest_score"],ascending=[True,False])
    selected=[]
    last={}
    scan_no=0
    for date,g in x.groupby("signal_date",sort=True):
        count=0
        for _,r in g.iterrows():
            tic=r["ticker"]
            if tic in last and scan_no-last[tic]<=int(cooldown_scans):
                continue
            selected.append(r.to_dict())
            last[tic]=scan_no
            count+=1
            if count>=int(top_n_per_scan):
                break
        scan_no += 1
    return pd.DataFrame(selected)

def summarize_signals(signals,horizon=60,use_net=True):
    empty={"signals":0,"avg_return":0.0,"median_return":0.0,"avg_spy":0.0,"avg_excess":0.0,
           "positive_rate":0.0,"beat_spy_rate":0.0,"best":0.0,"worst":0.0,
           "ci95_low":0.0,"ci95_high":0.0,"delist_events":0}
    if signals is None or signals.empty:
        return empty

    h=int(horizon)
    rc=f"future_{h}d_net_pct" if use_net and f"future_{h}d_net_pct" in signals else f"future_{h}d_pct"
    sc=f"spy_future_{h}d_net_pct" if use_net and f"spy_future_{h}d_net_pct" in signals else f"spy_future_{h}d_pct"
    ret=pd.to_numeric(signals[rc],errors="coerce").dropna()
    if ret.empty:
        return empty
    spy=pd.to_numeric(signals.loc[ret.index,sc],errors="coerce")
    excess=ret-spy
    n=len(excess)
    mean=float(excess.mean())
    se=float(excess.std(ddof=1)/np.sqrt(n)) if n>1 else 0.0
    dcol=f"delist_crossed_{h}d"
    delists=int(signals.loc[ret.index,dcol].fillna(False).sum()) if dcol in signals else 0

    return {
        "signals":int(n),"avg_return":float(ret.mean()),"median_return":float(ret.median()),
        "avg_spy":float(spy.mean()),"avg_excess":mean,
        "positive_rate":float((ret>0).mean()*100),
        "beat_spy_rate":float((excess>0).mean()*100),
        "best":float(ret.max()),"worst":float(ret.min()),
        "ci95_low":mean-1.96*se,"ci95_high":mean+1.96*se,
        "delist_events":delists,
    }

def score_buckets(signals,horizon=60):
    if signals is None or signals.empty:
        return pd.DataFrame()
    x=signals.copy()
    x["score_bucket"]=pd.cut(
        x["backtest_score"],bins=[0,40,50,60,70,80,90,100],include_lowest=True
    ).astype(str)
    h=int(horizon)
    rc=f"future_{h}d_net_pct" if f"future_{h}d_net_pct" in x else f"future_{h}d_pct"
    out=x.groupby("score_bucket",observed=False).agg(
        segnali=("ticker","count"),
        rendimento_medio=(rc,"mean"),
        excess_medio=(f"excess_{h}d_pct","mean"),
        score_medio=("backtest_score","mean")
    ).reset_index()
    return out[out["segnali"]>0]
