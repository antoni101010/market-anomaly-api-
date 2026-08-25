import math
import pandas as pd

class DemoPointInTimeFundamentals:
    def metrics_as_of(self, ticker, as_of, last_price=None):
        d=pd.Timestamp(as_of)
        seed=sum(ord(c) for c in ticker.upper())
        cycle=math.sin((d.year*12+d.month+seed)/8.0)
        growth=max(-20,min(45,10+(seed%13)+cycle*10))
        margin=max(-15,min(45,8+(seed%17)+cycle*5))
        leverage=max(0.25,min(0.95,0.45+(seed%20)/100-cycle*0.08))
        fcf=max(-12,min(40,margin-3+cycle*4))
        filed=(d-pd.Timedelta(days=45+(seed%35))).date().isoformat()
        return {"revenue_growth_pct":growth,"net_margin_pct":margin,"liabilities_to_assets":leverage,
                "fcf_margin_pct":fcf,"pit_last_filed_date":filed,"pit_fact_count":10}
