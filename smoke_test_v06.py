import pandas as pd
from providers.demo import DemoProvider
from providers.pit_demo import DemoPointInTimeFundamentals
from backtest import build_feature_dataset,select_signals,summarize_signals
from walkforward import walk_forward_validate
from model import BACKTEST_WEIGHTS
u=pd.read_csv('universe.csv').head(8)
ds=build_feature_dataset(u,DemoProvider(),years=3,scan_every=20,fundamental_provider=DemoPointInTimeFundamentals(),commission_bps=5,slippage_bps=10)
assert len(ds)>0 and ds['pit_data_available'].mean()>0.9
s=select_signals(ds,BACKTEST_WEIGHTS,threshold=35,top_n_per_scan=2,cooldown_scans=1)
m=summarize_signals(s,60)
assert 'ci95_low' in m and 'future_60d_net_pct' in s.columns
wf=walk_forward_validate(ds,60,train_years=1,test_months=6,iterations=10,threshold=35,top_n=2,cooldown=1)
assert 'reliability_grade' in wf
print('OK',len(ds),len(s),m['avg_excess'],wf['reliability_grade'])
