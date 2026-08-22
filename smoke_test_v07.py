
import pandas as pd
from providers.demo import DemoProvider
from providers.pit_demo import DemoPointInTimeFundamentals
from universe_manager import build_demo_historical_universe, active_snapshot, universe_summary
from backtest import build_feature_dataset, select_signals, summarize_signals
from bias_audit import run_bias_audit
from model import BACKTEST_WEIGHTS

u=build_demo_historical_universe(45,seed=11)
summary=universe_summary(u)
assert summary["symbols"]==45
assert summary["delisted_symbols"]>0

snap=active_snapshot(u,"2020-06-30")
assert 0 < len(snap) <= len(u)

ds=build_feature_dataset(
    u,DemoProvider(),years=3,scan_every=20,
    fundamental_provider=DemoPointInTimeFundamentals(),
    commission_bps=5,slippage_bps=10
)
assert len(ds)>0
assert "active_universe_size" in ds.columns
assert "history_coverage_pct" in ds.columns
assert "delist_crossed_60d" in ds.columns

s=select_signals(ds,BACKTEST_WEIGHTS,threshold=30,top_n_per_scan=3,cooldown_scans=1)
m=summarize_signals(s,60)
assert "delist_events" in m

audit=run_bias_audit(u,ds,"all",True,5,10)
assert audit.attrs["audit_grade"] in list("ABCDE")

print("OK V0.7")
print("Universe:",summary)
print("Dataset rows:",len(ds))
print("Signals:",len(s))
print("60d excess:",round(m["avg_excess"],3))
print("Delisting events evaluated:",m["delist_events"])
print("Audit:",audit.attrs["audit_grade"],audit.attrs["audit_score_pct"])
