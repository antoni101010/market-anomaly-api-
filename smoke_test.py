import pandas as pd
from providers.demo import DemoProvider
from backtest import build_feature_dataset, select_signals, summarize_signals
from optimizer import optimize_weights

u = pd.read_csv("universe.csv").head(10)
p = DemoProvider()
ds = build_feature_dataset(u, p, years=3, scan_every=20)
assert not ds.empty
sig = select_signals(ds, threshold=40, top_n_per_scan=3, cooldown_scans=1)
assert not sig.empty
metrics = summarize_signals(sig, 60)
opt = optimize_weights(
    ds, horizon=60, iterations=30, threshold=40,
    top_n_per_scan=3, cooldown_scans=1, train_ratio=0.70
)
assert abs(sum(opt["weights"].values()) - 1.0) < 1e-6
print("OK")
print("feature_rows:", len(ds))
print("signals:", len(sig))
print("60d_metrics:", metrics)
print("holdout_metrics:", opt["test_metrics"])
