from __future__ import annotations
import os
from datetime import datetime
import pandas as pd
import streamlit as st

from config import CONFIG
from scanner import scan_universe
from providers.twelve_data import TwelveDataProvider
from providers.demo import DemoProvider
from providers.pit_demo import DemoPointInTimeFundamentals
from providers.sec_edgar import SecEdgarProvider
from providers.alpha_vantage_universe import AlphaVantageUniverseProvider
from universe_manager import normalize_universe, build_demo_historical_universe, active_snapshot, universe_summary
from backtest import build_feature_dataset, select_signals, summarize_signals, score_buckets
from optimizer import optimize_weights
from walkforward import walk_forward_validate
from bias_audit import run_bias_audit
from statistics_engine import bootstrap_excess, sign_flip_pvalue, group_analysis, cohort_curve, weight_stability
from model import BACKTEST_WEIGHTS
from storage import save_signals, load_signals, add_alert_rule, list_alert_rules, delete_alert_rule
from alerts import evaluate_alerts

st.set_page_config(page_title=CONFIG.app_name,page_icon="📉",layout="wide")
st.title(CONFIG.app_name)
st.caption("v2.2 — ricerca statistica su anomalie, eventi, backtest point-in-time e validazione")

for k in ["live_results","bt_dataset","bt_signals","opt_result","wf_result","bt_params","alpha_universe","custom_universe"]:
    if k not in st.session_state: st.session_state[k]=None

with st.sidebar:
    st.header("Dati di mercato")
    price_mode=st.radio("Prezzi",["Demo","Twelve Data"],index=0)
    twelve_key=st.text_input("Twelve Data API key",type="password") if price_mode=="Twelve Data" else ""
    batch_size=st.slider("Batch",5,50,25,5)
    st.header("Universo")
    universe_mode=st.radio("Fonte",["Demo storico","CSV storico","Alpha Vantage lifecycle"],index=0)
    alpha_key=st.text_input("Alpha Vantage API key",type="password") if universe_mode=="Alpha Vantage lifecycle" else ""
    st.header("Scanner live")
    live_limit=st.slider("Titoli",10,1000,100,10)
    catalyst_top_n=st.slider("Catalyst sui migliori",1,25,7)
    min_live=st.slider("Somiglianza storica minima",0,100,55)
    max_trap=st.slider("Value Trap massimo",0,100,65)
    run_live=st.button("Scansiona ora",type="primary",use_container_width=True)

def market_provider():
    return DemoProvider() if price_mode=="Demo" else TwelveDataProvider(twelve_key,batch_size=batch_size,cache_dir=CONFIG.price_cache_dir)

def pit_provider():
    return DemoPointInTimeFundamentals() if price_mode=="Demo" else SecEdgarProvider()

# Universe
if universe_mode=="Demo storico":
    universe=build_demo_historical_universe(250)
elif universe_mode=="CSV storico":
    up=st.sidebar.file_uploader("CSV storico",type=["csv"])
    if up is not None:
        try: st.session_state.custom_universe=normalize_universe(pd.read_csv(up),source="uploaded_csv")
        except Exception as e: st.sidebar.error(str(e))
    universe=st.session_state.custom_universe if st.session_state.custom_universe is not None else build_demo_historical_universe(250)
else:
    if st.sidebar.button("Importa lifecycle USA",use_container_width=True):
        if not alpha_key: st.sidebar.error("Inserisci API key Alpha Vantage")
        else:
            try:
                st.session_state.alpha_universe=AlphaVantageUniverseProvider(alpha_key).build_lifecycle_universe(False,["NYSE","NASDAQ","AMEX"])
            except Exception as e: st.sidebar.error(str(e))
    universe=st.session_state.alpha_universe if st.session_state.alpha_universe is not None else build_demo_historical_universe(250)
universe=normalize_universe(universe)

if run_live:
    if price_mode=="Twelve Data" and not twelve_key:
        st.error("Inserisci la API key Twelve Data.")
    else:
        live_u=active_snapshot(universe,pd.Timestamp.today()).head(live_limit)
        with st.spinner(f"Analizzo {len(live_u)} titoli..."):
            st.session_state.live_results=scan_universe(live_u,market_provider(),include_sec=(price_mode=="Twelve Data"),catalyst_top_n=catalyst_top_n)

tabs=st.tabs(["Dashboard","Scanner","Catalyst","Backtest","Statistica","Walk-forward","Universo & Bias","Alert","Storico","Setup commerciale"])

with tabs[0]:
    st.subheader("Dashboard")
    df=st.session_state.live_results
    if df is None:
        st.info("Esegui una scansione live. In modalità Demo non servono API key.")
    else:
        valid=df[df['error'].isna()].copy()
        candidates=valid[(valid['opportunity_score']>=min_live)&(valid['value_trap_risk']<=max_trap)].copy()
        a,b,c,d=st.columns(4)
        a.metric("Analizzati",len(valid)); b.metric("Candidati",len(candidates))
        c.metric("Somiglianza storica max",f"{valid['opportunity_score'].max():.1f}" if len(valid) else "-")
        d.metric("Anomaly max",f"{valid['anomaly_score'].max():.1f}" if len(valid) else "-")
        if len(candidates):
            st.dataframe(candidates[["ticker","company","opportunity_score","anomaly_score","value_trap_risk","catalyst_risk","quality_score","catalyst_label"]].head(20).rename(columns={"opportunity_score":"somiglianza_casi_storici"}),use_container_width=True,hide_index=True)
            hits=evaluate_alerts(valid)
            if not hits.empty: st.success(f"{len(hits)} alert attivi scattati.")

with tabs[1]:
    st.subheader("Top anomalie")
    df=st.session_state.live_results
    if df is None: st.info("Esegui la scansione live.")
    else:
        valid=df[df['error'].isna()].copy()
        filt=valid[(valid['opportunity_score']>=min_live)&(valid['value_trap_risk']<=max_trap)].copy()
        st.dataframe(filt[["ticker","company","opportunity_score","anomaly_score","recovery_potential","value_trap_risk","catalyst_risk","quality_score","drawdown_52w_pct","relative_60d_vs_spy_pct"]].rename(columns={"opportunity_score":"somiglianza_casi_storici","recovery_potential":"pattern_recupero_storico"}),use_container_width=True,hide_index=True)
        if len(filt):
            tic=st.selectbox("Scheda",filt['ticker'].tolist(),key="scan_ticker")
            r=filt[filt.ticker==tic].iloc[0]
            a,b,c,d=st.columns(4)
            a.metric("Casi storici",f"{r.opportunity_score:.1f}/100"); b.metric("Anomaly",f"{r.anomaly_score:.1f}/100")
            c.metric("Value Trap",f"{r.value_trap_risk:.1f}/100"); d.metric("Catalyst Risk",f"{r.catalyst_risk:.1f}/100")
            st.write(r.explanation); st.info(f"{r.catalyst_label}: {r.catalyst_explanation}")
            if st.button("Salva candidati",key="save_live"): st.success(f"Salvati {save_signals(filt)} segnali")
            st.download_button("Esporta CSV",filt.to_csv(index=False).encode(),"market_anomaly_live.csv","text/csv")

with tabs[2]:
    st.subheader("Catalyst Lab")
    df=st.session_state.live_results
    if df is None: st.info("Esegui prima una scansione.")
    else:
        ana=df[(df['error'].isna())&(df['catalyst_label']!='Non analizzato')]
        for _,r in ana.iterrows():
            with st.expander(f"{r.ticker} — {r.catalyst_label} — rischio {r.catalyst_risk:.0f}/100"):
                st.write(r.catalyst_explanation)
                for item in (r.get('catalyst_items') or []): st.write(f"• {item.get('datetime','')} {item.get('title','')}")
                for f in (r.get('recent_filings') or []): st.write(f"• SEC {f.get('filing_date','')} — {f.get('form','')}")

with tabs[3]:
    st.subheader("Backtest point-in-time")
    c1,c2,c3,c4=st.columns(4)
    years=c1.slider("Anni",2,12,5,key="bt_years"); scan_every=c2.select_slider("Ogni sedute",[5,10,15,20],value=10)
    horizon=c3.selectbox("Orizzonte",[20,60,90],index=1); threshold=c4.slider("Score minimo",20,90,50,key="bt_threshold")
    c1,c2,c3,c4=st.columns(4)
    top_n=c1.slider("Top per data",1,10,3); cooldown=c2.slider("Cooldown",0,8,2)
    commission=c3.number_input("Commissione bps/lato",0.0,50.0,5.0); slippage=c4.number_input("Slippage bps/lato",0.0,100.0,10.0)
    use_pit=st.checkbox("Fondamentali point-in-time",value=True)
    max_bt=max(20,min(1500,len(universe))); default_bt=min(250,max_bt)
    bt_limit=st.slider("Titoli nel backtest",20,max_bt,default_bt,10)
    if st.button("Esegui backtest",type="primary",key="run_bt"):
        if price_mode=="Twelve Data" and not twelve_key: st.error("Serve la API key Twelve Data.")
        else:
            bt_u=universe.head(bt_limit)
            with st.spinner("Costruisco il dataset storico senza usare dati futuri..."):
                ds=build_feature_dataset(bt_u,market_provider(),years,scan_every,fundamental_provider=pit_provider() if use_pit else None,adjust="all",commission_bps=commission,slippage_bps=slippage)
                sig=select_signals(ds,BACKTEST_WEIGHTS,threshold,top_n,cooldown)
                st.session_state.bt_dataset=ds; st.session_state.bt_signals=sig
                st.session_state.bt_params={"horizon":horizon,"threshold":threshold,"top_n":top_n,"cooldown":cooldown,"commission":commission,"slippage":slippage,"use_pit":use_pit,"universe":bt_u}
    sig=st.session_state.bt_signals
    if sig is not None:
        h=st.session_state.bt_params['horizon']; m=summarize_signals(sig,h)
        a,b,c,d,e=st.columns(5)
        a.metric("Segnali",m['signals']); b.metric("Rendimento netto",f"{m['avg_return']:.2f}%"); c.metric("Excess vs SPY",f"{m['avg_excess']:.2f}%")
        d.metric("Batte SPY",f"{m['beat_spy_rate']:.1f}%"); e.metric("Delisting",m.get('delist_events',0))
        st.dataframe(score_buckets(sig,h),use_container_width=True,hide_index=True)
        st.download_button("Scarica backtest",sig.to_csv(index=False).encode(),"market_anomaly_backtest.csv","text/csv")

with tabs[4]:
    st.subheader("Validazione statistica")
    sig=st.session_state.bt_signals
    if sig is None or sig.empty: st.info("Esegui prima il backtest.")
    else:
        h=st.session_state.bt_params['horizon']
        boot=bootstrap_excess(sig,h,3000); p=sign_flip_pvalue(sig,h,5000)
        curve,cs=cohort_curve(sig,h)
        a,b,c,d=st.columns(4)
        a.metric("Excess medio",f"{boot['mean_excess']:.2f}%"); b.metric("Bootstrap CI95",f"{boot['ci95_low']:.2f} / {boot['ci95_high']:.2f}%")
        c.metric("P(excess > 0)",f"{boot['prob_gt_zero']:.1f}%"); d.metric("Permutation p-value",f"{p:.4f}")
        if cs:
            a,b,c=st.columns(3); a.metric("Cohort total",f"{cs['strategy_total_pct']:.1f}%"); b.metric("Benchmark cohort",f"{cs['benchmark_total_pct']:.1f}%"); c.metric("Max drawdown",f"{cs['max_drawdown_pct']:.1f}%")
            st.line_chart(curve.set_index('signal_date')[["strategy_equity","benchmark_equity"]])
        st.markdown("#### Per regime di mercato")
        st.dataframe(group_analysis(sig,'market_regime',h),use_container_width=True,hide_index=True)
        st.markdown("#### Per settore")
        st.dataframe(group_analysis(sig,'sector_etf',h),use_container_width=True,hide_index=True)
        if p < 0.05 and boot['ci95_low']>0: st.success("Il vantaggio osservato supera i controlli statistici base di questa versione.")
        else: st.warning("Il vantaggio osservato NON è ancora abbastanza robusto da considerarlo dimostrato.")

with tabs[5]:
    st.subheader("Walk-forward")
    ds=st.session_state.bt_dataset
    if ds is None: st.info("Esegui prima il backtest.")
    else:
        c1,c2,c3=st.columns(3)
        tr=c1.slider("Train anni",1,5,3); tm=c2.selectbox("Test mesi",[3,6,12],index=1); it=c3.slider("Tentativi/fold",20,250,80,20)
        if st.button("Esegui walk-forward"):
            p=st.session_state.bt_params
            st.session_state.wf_result=walk_forward_validate(ds,p['horizon'],tr,tm,it,p['threshold'],p['top_n'],p['cooldown'])
        wf=st.session_state.wf_result
        if wf:
            m=wf['overall']; wt,stab=weight_stability(wf['folds'])
            a,b,c,d=st.columns(4); a.metric("Affidabilità",wf['reliability_grade']); b.metric("Excess OOS",f"{m['avg_excess']:.2f}%"); c.metric("Fold positivi",f"{wf['positive_folds_pct']:.1f}%"); d.metric("Stabilità pesi",f"{stab:.0f}/100")
            st.dataframe(wf['folds'],use_container_width=True,hide_index=True); st.dataframe(wt,use_container_width=True,hide_index=True)

with tabs[6]:
    st.subheader("Universo storico e Bias Audit")
    us=universe_summary(universe)
    a,b,c,d=st.columns(4); a.metric("Simboli",us['symbols']); b.metric("Lifecycle",f"{us['lifecycle_coverage_pct']:.1f}%"); c.metric("Delistati",us['delisted_symbols']); d.metric("Terminal return",f"{us['terminal_return_coverage_pct']:.1f}%")
    date=st.date_input("Snapshot storico",value=pd.Timestamp('2022-06-30').date()); snap=active_snapshot(universe,date)
    st.dataframe(snap[[c for c in ['ticker','company','sector_etf','active_from','active_to','delisting_return_pct','universe_source'] if c in snap.columns]].head(1000),use_container_width=True,hide_index=True)
    p=st.session_state.bt_params or {"use_pit":False,"commission":5,"slippage":10,"universe":universe}
    audit=run_bias_audit(p.get('universe',universe),st.session_state.bt_dataset,'all',p.get('use_pit',False),p.get('commission',5),p.get('slippage',10))
    a,b=st.columns(2); a.metric("Bias grade",audit.attrs.get('audit_grade','-')); b.metric("Audit score",f"{audit.attrs.get('audit_score_pct',0):.1f}%")
    st.dataframe(audit,use_container_width=True,hide_index=True)

with tabs[7]:
    st.subheader("Alert Center")
    with st.form("new_alert"):
        name=st.text_input("Nome regola",value="Top anomaly")
        a,b,c,d=st.columns(4)
        mo=a.number_input("Somiglianza storica ≥",0,100,75); ma=b.number_input("Anomaly ≥",0,100,60); vt=c.number_input("Value Trap ≤",0,100,50); cr=d.number_input("Catalyst Risk ≤",0,100,60)
        if st.form_submit_button("Salva regola"):
            add_alert_rule(name,mo,ma,vt,cr); st.success("Regola salvata")
    rules=list_alert_rules(); st.dataframe(rules,use_container_width=True,hide_index=True)
    if not rules.empty:
        rid=st.selectbox("Elimina regola",rules['id'].tolist(),format_func=lambda x: f"{x} — {rules[rules.id==x].iloc[0]['name']}")
        if st.button("Elimina",key="del_rule"): delete_alert_rule(rid); st.success("Regola eliminata")
    if st.session_state.live_results is not None:
        hits=evaluate_alerts(st.session_state.live_results,rules); st.markdown("#### Alert scattati"); st.dataframe(hits,use_container_width=True,hide_index=True)

with tabs[8]:
    st.subheader("Storico segnali")
    hist=load_signals(); st.dataframe(hist,use_container_width=True,hide_index=True)

with tabs[9]:
    st.subheader("Setup commerciale")
    st.markdown("""
La v2.2 include il livello legale nell'app mobile, la metodologia, la tracciatura versionata dell'accettazione e il Global Market Tension Engine. Prima di una vendita pubblica restano da verificare gli elementi esterni al codice: licenza commerciale dei dati, hosting/dominio, eventuale autenticazione e pagamenti e revisione professionale del modello commerciale concreto.

Il prodotto è progettato come **strumento di ricerca statistica**. Non determina l'adeguatezza di uno strumento per una persona, non gestisce portafogli, non esegue ordini e non formula istruzioni personalizzate di acquisto o vendita.
""")
    st.code("streamlit run app.py")
    st.caption("Per SEC imposta SEC_USER_AGENT con nome progetto + email di contatto reale.")

st.divider()
st.caption("Market Anomaly v2.2 — ricerca statistica. Dati e risultati storici possono essere incompleti o ritardati e non garantiscono risultati futuri.")
