import pandas as pd
from providers.demo import DemoProvider
from providers.pit_demo import DemoPointInTimeFundamentals
from universe_manager import build_demo_historical_universe
from scanner import scan_universe
from backtest import build_feature_dataset,select_signals,summarize_signals
from statistics_engine import bootstrap_excess,sign_flip_pvalue,group_analysis,cohort_curve
from bias_audit import run_bias_audit
from model import BACKTEST_WEIGHTS
from alerts import evaluate_alerts


def test_full_core():
    u=build_demo_historical_universe(50,seed=17)
    live=scan_universe(u.head(20),DemoProvider(),include_sec=False,catalyst_top_n=5)
    assert live['opportunity_score'].notna().sum() >= 15

    ds=build_feature_dataset(u,DemoProvider(),years=3,scan_every=20,fundamental_provider=DemoPointInTimeFundamentals(),commission_bps=5,slippage_bps=10)
    assert len(ds)>100
    assert {'market_regime','spy_trailing_60d_pct','active_universe_size'}.issubset(ds.columns)

    sig=select_signals(ds,BACKTEST_WEIGHTS,threshold=30,top_n_per_scan=3,cooldown_scans=1)
    assert len(sig)>10
    m=summarize_signals(sig,60)
    assert m['signals']>0

    b=bootstrap_excess(sig,60,n_boot=300)
    assert b['n']>0 and 0<=b['prob_gt_zero']<=100
    p=sign_flip_pvalue(sig,60,n_perm=300)
    assert 0<=p<=1
    assert not group_analysis(sig,'market_regime',60,min_n=2).empty
    curve,stats=cohort_curve(sig,60)
    assert not curve.empty and 'max_drawdown_pct' in stats

    audit=run_bias_audit(u,ds,'all',True,5,10)
    assert audit.attrs['audit_grade'] in list('ABCDE')
