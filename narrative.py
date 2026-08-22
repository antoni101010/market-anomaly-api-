"""
Genera le spiegazioni testuali per la scheda di dettaglio titolo:
- "Perché potrebbe essere un'anomalia"
- "Perché potrebbe NON esserlo" (rischi, per non creare falsi segnali ottimistici)
- Classificazione dell'Opportunity Score in fasce leggibili (soglie configurabili)

Questo modulo non introduce nuovi calcoli statistici: riusa solo i valori
già prodotti da scanner.py / model.py / catalyst_engine.py e li traduce
in un linguaggio comprensibile anche a chi non è un analista finanziario.
"""
from __future__ import annotations
from dataclasses import dataclass


DEFAULT_THRESHOLDS = {
    "exceptional": 90,   # 90-100 = ANOMALIA ECCEZIONALE
    "strong": 80,        # 80-89  = ANOMALIA FORTE
    "interesting": 70,   # 70-79  = INTERESSANTE
    "watch": 60,         # 60-69  = DA MONITORARE
    # sotto 60             = NON PRIORITARIA
}


def classify_opportunity(score: float, thresholds: dict | None = None) -> dict:
    t = thresholds or DEFAULT_THRESHOLDS
    score = float(score) if score is not None else 0.0
    if score >= t["exceptional"]:
        label, key = "ANOMALIA ECCEZIONALE", "exceptional"
    elif score >= t["strong"]:
        label, key = "ANOMALIA FORTE", "strong"
    elif score >= t["interesting"]:
        label, key = "INTERESSANTE", "interesting"
    elif score >= t["watch"]:
        label, key = "DA MONITORARE", "watch"
    else:
        label, key = "NON PRIORITARIA", "low"
    return {"label": label, "key": key, "score": score}


def why_it_might_be_anomaly(row: dict) -> list[str]:
    """Argomenti a favore: il crollo sembra un'esagerazione del mercato."""
    reasons = []
    dd = row.get("drawdown_52w_pct")
    if dd is not None and dd <= -25:
        reasons.append(
            f"Il titolo ha perso circa {abs(dd):.0f}% dal massimo delle ultime 52 settimane: "
            "un calo di questa entità è raro e spesso segnala una reazione emotiva più che un "
            "cambiamento graduale della situazione aziendale."
        )
    rel_spy = row.get("relative_60d_vs_spy_pct")
    if rel_spy is not None and rel_spy <= -12:
        reasons.append(
            f"Negli ultimi 60 giorni il titolo ha fatto circa {abs(rel_spy):.0f} punti percentuali "
            "peggio del mercato generale (SPY): non è solo un calo di mercato, è specifico del titolo."
        )
    vol = row.get("volume_ratio_20d")
    if vol is not None and vol >= 1.8:
        reasons.append(
            "I volumi di scambio sono molto più alti del normale, segno che molti investitori "
            "hanno reagito nello stesso momento — tipico delle vendite indiscriminate."
        )
    rsi = row.get("rsi14")
    if rsi is not None and rsi <= 32:
        reasons.append(
            "L'indicatore RSI è in zona di forte ipervenduto: il titolo è stato venduto in modo "
            "intenso e rapido rispetto alla sua storia recente."
        )
    quality = row.get("quality_score")
    if quality is not None and quality >= 65:
        reasons.append(
            f"I fondamentali disponibili restano relativamente solidi (qualità stimata {quality:.0f}/100), "
            "il che rende meno probabile che il crollo rifletta un deterioramento strutturale dell'azienda."
        )
    catalyst_label = row.get("catalyst_label")
    if catalyst_label == "Catalizzatore negativo potenzialmente temporaneo":
        reasons.append(
            "L'evento scatenante individuato (guidance, stime mancate, fattori operativi) sembra "
            "avere natura temporanea più che strutturale."
        )
    if not reasons:
        reasons.append(
            "Il punteggio complessivo indica un'anomalia moderata: nessun singolo fattore domina, "
            "ma la combinazione di elementi tecnici giustifica comunque l'attenzione."
        )
    return reasons


def why_it_might_not_be(row: dict) -> list[str]:
    """Argomenti di cautela: motivi per cui il crollo potrebbe essere giustificato."""
    risks = []
    trap = row.get("value_trap_risk")
    if trap is not None and trap >= 65:
        risks.append(
            f"Il rischio di 'value trap' è elevato ({trap:.0f}/100): il prezzo basso potrebbe "
            "riflettere problemi reali e persistenti, non un'occasione."
        )
    catalyst_label = row.get("catalyst_label")
    catalyst_risk = row.get("catalyst_risk")
    if catalyst_label == "Possibile rischio strutturale":
        risks.append(
            "Il motore ha rilevato termini associati a rischi strutturali (es. problemi di bilancio, "
            "indagini, liquidità, delisting): questi casi richiedono massima cautela."
        )
    elif catalyst_risk is not None and catalyst_risk >= 60:
        risks.append(
            f"Il rischio legato all'evento scatenante è comunque elevato ({catalyst_risk:.0f}/100)."
        )
    quality = row.get("quality_score")
    if quality is not None and quality < 45:
        risks.append(
            f"I fondamentali disponibili sono deboli (qualità stimata {quality:.0f}/100): "
            "il calo potrebbe essere una correzione giustificata, non un'esagerazione."
        )
    liab = row.get("liabilities_to_assets")
    if liab is not None and liab >= 0.7:
        risks.append(
            "Il livello di indebitamento rispetto agli attivi è elevato, il che aumenta il rischio "
            "in caso di ulteriore deterioramento del business."
        )
    rec_growth = row.get("revenue_growth_pct")
    if rec_growth is not None and rec_growth < 0:
        risks.append(
            f"La crescita dei ricavi risulta negativa ({rec_growth:.1f}%): un dato da monitorare "
            "con attenzione prima di considerare il calo un'esagerazione."
        )
    if row.get("error"):
        risks.append("Dati incompleti per questo titolo: trattare l'analisi con cautela.")
    if not risks:
        risks.append(
            "Non emergono segnali di rischio strutturale evidenti dai dati disponibili, ma questo "
            "non esclude rischi non catturati dal modello (es. eventi futuri, contesto macro)."
        )
    return risks


def build_ticker_narrative(row: dict, thresholds: dict | None = None) -> dict:
    classification = classify_opportunity(row.get("opportunity_score", 0), thresholds)
    return {
        "classification": classification,
        "why_anomaly": why_it_might_be_anomaly(row),
        "why_not": why_it_might_not_be(row),
    }
