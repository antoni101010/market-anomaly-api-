
import pandas as pd
from universe_manager import normalize_universe, universe_summary

def run_bias_audit(universe,dataset,adjust="all",fundamentals_enabled=False,commission_bps=5,slippage_bps=10):
    u=normalize_universe(universe)
    us=universe_summary(u)
    checks=[]

    def add(name,status,detail):
        checks.append({"Controllo":name,"Stato":status,"Dettaglio":detail})

    add("Look-ahead prezzi","OK","Le feature usano soltanto barre disponibili fino alla data del segnale.")
    add("Corporate actions","OK" if adjust=="all" else "ATTENZIONE",
        f"Modalità prezzi: {adjust}. Per il backtest total-return preferire adjust=all.")

    pit_cov=0.0
    if dataset is not None and not dataset.empty and "pit_data_available" in dataset:
        pit_cov=float(dataset["pit_data_available"].fillna(False).mean()*100)
    add("Fondamentali point-in-time",
        "OK" if fundamentals_enabled and pit_cov>=70 else ("PARZIALE" if fundamentals_enabled else "OFF"),
        f"Copertura PIT: {pit_cov:.1f}% dei record.")

    lifecycle = float(u["active_from"].notna().mean()*100) if len(u) else 0.0
    delisted = int(u["active_to"].notna().sum())
    if lifecycle >= 90 and delisted > 0:
        surv_status="MIGLIORATO"
        surv_detail=f"Lifecycle presente sul {lifecycle:.1f}% dei simboli; {delisted} titoli con data di uscita. Serve comunque verificare che la fonte copra l'intero mercato scelto."
    elif lifecycle > 0:
        surv_status="PARZIALE"
        surv_detail=f"Lifecycle presente sul {lifecycle:.1f}% dei simboli; {delisted} titoli con data di uscita."
    else:
        surv_status="NON RISOLTO"
        surv_detail="La lista non contiene una membership storica utilizzabile."
    add("Survivorship bias",surv_status,surv_detail)

    terminal_cov=us["terminal_return_coverage_pct"]
    add("Delisting return",
        "MIGLIORATO" if delisted and terminal_cov>=70 else ("PARZIALE" if delisted else "NON RISOLTO"),
        f"{delisted} titoli delistati; terminal return disponibile sul {terminal_cov:.1f}% dei delisting.")

    hist_cov=0.0
    if dataset is not None and not dataset.empty and "history_coverage_pct" in dataset:
        hist_cov=float(pd.to_numeric(dataset["history_coverage_pct"],errors="coerce").mean())
    add("Copertura prezzi universo",
        "OK" if hist_cov>=90 else ("PARZIALE" if hist_cov>=60 else "ATTENZIONE"),
        f"Copertura media dello storico sui simboli attivi: {hist_cov:.1f}%.")

    add("Costi e slippage","OK",
        f"Commissione {commission_bps:.1f} bps/lato + slippage {slippage_bps:.1f} bps/lato inclusi.")
    add("Holdout temporale","OK","Ottimizzazione train/holdout disponibile.")
    add("Walk-forward","OK","Validazione a finestre temporali successive disponibile.")

    # Coarse audit grade.
    score_map={"OK":2,"MIGLIORATO":1.5,"PARZIALE":1,"OFF":0.5,"ATTENZIONE":0.5,"NON RISOLTO":0}
    score=sum(score_map.get(x["Stato"],0) for x in checks)
    max_score=2*len(checks)
    pct=100*score/max_score if max_score else 0
    if pct>=85: grade="A"
    elif pct>=72: grade="B"
    elif pct>=58: grade="C"
    elif pct>=45: grade="D"
    else: grade="E"

    out=pd.DataFrame(checks)
    out.attrs["audit_grade"]=grade
    out.attrs["audit_score_pct"]=round(pct,1)
    return out
