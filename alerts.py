from __future__ import annotations
import pandas as pd
from storage import list_alert_rules

def evaluate_alerts(results: pd.DataFrame, rules: pd.DataFrame | None = None) -> pd.DataFrame:
    if results is None or results.empty:
        return pd.DataFrame()
    rules = list_alert_rules() if rules is None else rules
    if rules is None or rules.empty:
        return pd.DataFrame()
    hits=[]
    valid=results[results.get('error').isna()].copy() if 'error' in results.columns else results.copy()
    for _, rule in rules[rules['enabled'].astype(int).eq(1)].iterrows():
        x=valid[
            (pd.to_numeric(valid['opportunity_score'],errors='coerce') >= float(rule['min_opportunity'])) &
            (pd.to_numeric(valid['anomaly_score'],errors='coerce') >= float(rule['min_anomaly'])) &
            (pd.to_numeric(valid['value_trap_risk'],errors='coerce') <= float(rule['max_value_trap'])) &
            (pd.to_numeric(valid['catalyst_risk'],errors='coerce') <= float(rule['max_catalyst_risk']))
        ].copy()
        for _, r in x.iterrows():
            hits.append({
                'rule_id':int(rule['id']),'rule_name':rule['name'],'ticker':r['ticker'],'company':r.get('company',''),
                'opportunity_score':r['opportunity_score'],'anomaly_score':r['anomaly_score'],
                'value_trap_risk':r['value_trap_risk'],'catalyst_risk':r['catalyst_risk'],
                'catalyst_label':r.get('catalyst_label','')
            })
    return pd.DataFrame(hits).sort_values(['opportunity_score','anomaly_score'],ascending=False) if hits else pd.DataFrame()
