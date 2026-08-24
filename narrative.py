"""
Genera le spiegazioni testuali per la scheda
di dettaglio del titolo:

- Perché potrebbe essere un'anomalia
- Perché potrebbe non esserlo
- Classificazione dell'Opportunity Score

Il modulo traduce i risultati del motore
in un linguaggio semplice e comprensibile.

Non genera consigli di acquisto o vendita.
"""

from __future__ import annotations


DEFAULT_THRESHOLDS = {
    "exceptional": 90,
    "strong": 80,
    "interesting": 70,
    "watch": 60,
}


def classify_opportunity(
    score: float,
    thresholds: dict | None = None,
) -> dict:

    limits = (
        thresholds
        or DEFAULT_THRESHOLDS
    )

    score = (
        float(score)
        if score is not None
        else 0.0
    )

    if score >= limits["exceptional"]:
        label = "ANOMALIA ECCEZIONALE"
        key = "exceptional"

    elif score >= limits["strong"]:
        label = "ANOMALIA FORTE"
        key = "strong"

    elif score >= limits["interesting"]:
        label = "POSSIBILE ANOMALIA"
        key = "interesting"

    elif score >= limits["watch"]:
        label = "DA APPROFONDIRE"
        key = "watch"

    else:
        label = "MOVIMENTO NON PRIORITARIO"
        key = "low"

    return {
        "label": label,
        "key": key,
        "score": score,
    }


def why_it_might_be_anomaly(
    row: dict,
) -> list[str]:
    """
    Elementi che possono indicare
    una reazione eccessiva del mercato.
    """

    reasons = []

    drawdown = row.get(
        "drawdown_52w_pct"
    )

    if (
        drawdown is not None
        and drawdown <= -25
    ):
        reasons.append(
            f"Il titolo ha perso circa "
            f"{abs(drawdown):.0f}% dal massimo "
            "delle ultime 52 settimane. "
            "Un calo di questa entità segnala "
            "un movimento particolarmente forte."
        )

    relative_market = row.get(
        "relative_60d_vs_spy_pct"
    )

    if (
        relative_market is not None
        and relative_market <= -12
    ):
        reasons.append(
            "Negli ultimi 60 giorni il titolo "
            f"ha fatto circa "
            f"{abs(relative_market):.0f} punti "
            "percentuali peggio del mercato. "
            "Il movimento non dipende solamente "
            "dall'andamento generale."
        )

    relative_sector = row.get(
        "relative_60d_vs_sector_pct"
    )

    if (
        relative_sector is not None
        and relative_sector <= -10
    ):
        reasons.append(
            "Il titolo ha avuto una performance "
            "molto peggiore rispetto alle altre "
            "società dello stesso settore."
        )

    volume = row.get(
        "volume_ratio_20d"
    )

    if (
        volume is not None
        and volume >= 1.8
    ):
        reasons.append(
            "I volumi di scambio sono molto "
            "più alti del normale. Molti "
            "investitori hanno reagito nello "
            "stesso momento."
        )

    rsi = row.get("rsi14")

    if (
        rsi is not None
        and rsi <= 32
    ):
        reasons.append(
            "La pressione di vendita è stata "
            "molto elevata e rapida rispetto "
            "alla storia recente del titolo."
        )

    quality = row.get(
        "quality_score"
    )

    if (
        quality is not None
        and quality >= 65
    ):
        reasons.append(
            "I fondamentali disponibili "
            "risultano relativamente solidi "
            f"(qualità stimata "
            f"{quality:.0f}/100)."
        )

    valuation = row.get(
        "valuation_score"
    )

    if (
        valuation is not None
        and valuation >= 65
    ):
        reasons.append(
            "Dopo il ribasso, la valutazione "
            "risulta relativamente contenuta "
            f"rispetto ai dati disponibili "
            f"({valuation:.0f}/100)."
        )

    catalyst_label = row.get(
        "catalyst_label"
    )

    if (
        catalyst_label
        == (
            "Catalizzatore negativo "
            "potenzialmente temporaneo"
        )
    ):
        reasons.append(
            "L'evento scatenante individuato "
            "sembra avere natura temporanea "
            "oppure operativa, anziché "
            "strutturale."
        )

    if not reasons:
        reasons.append(
            "Il motore ha individuato una "
            "combinazione di elementi insoliti, "
            "ma nessun singolo fattore risulta "
            "dominante."
        )

    return reasons


def why_it_might_not_be(
    row: dict,
) -> list[str]:
    """
    Elementi di cautela e motivi per cui
    il ribasso potrebbe essere giustificato.
    """

    risks = []

    value_trap = row.get(
        "value_trap_risk"
    )

    if (
        value_trap is not None
        and value_trap >= 65
    ):
        risks.append(
            "Il rischio di value trap è "
            f"elevato ({value_trap:.0f}/100). "
            "Il ribasso potrebbe riflettere "
            "problemi reali e persistenti."
        )

    valuation = row.get(
        "valuation_score"
    )

    if (
        valuation is not None
        and valuation < 40
    ):
        risks.append(
            "Nonostante il ribasso, la "
            "valutazione risulta ancora elevata "
            f"({valuation:.0f}/100). "
            "Il fatto che il prezzo sia sceso "
            "non significa automaticamente che "
            "il titolo sia diventato economico."
        )

    financial_risk = row.get(
        "financial_risk_score"
    )

    if (
        financial_risk is not None
        and financial_risk >= 65
    ):
        risks.append(
            "Il rischio finanziario risulta "
            f"elevato ({financial_risk:.0f}/100)."
        )

    distress_risk = row.get(
        "distress_risk_score"
    )

    if (
        distress_risk is not None
        and distress_risk >= 65
    ):
        risks.append(
            "Il rischio di difficoltà "
            "finanziaria risulta elevato "
            f"({distress_risk:.0f}/100)."
        )

    dilution_risk = row.get(
        "dilution_risk_score"
    )

    if (
        dilution_risk is not None
        and dilution_risk >= 65
    ):
        risks.append(
            "Il rischio di emissione di nuove "
            "azioni e conseguente diluizione "
            f"risulta elevato "
            f"({dilution_risk:.0f}/100)."
        )

    confidence = row.get(
        "confidence_score"
    )

    if (
        confidence is not None
        and confidence < 55
    ):
        risks.append(
            "L'analisi è incompleta "
            f"(affidabilità "
            f"{confidence:.0f}/100). "
            "Mancano alcuni dati necessari "
            "per una valutazione robusta."
        )

    catalyst_label = row.get(
        "catalyst_label"
    )

    catalyst_risk = row.get(
        "catalyst_risk"
    )

    if (
        catalyst_label
        == "Possibile rischio strutturale"
    ):
        risks.append(
            "Il motore ha rilevato elementi "
            "associati a possibili rischi "
            "strutturali, come problemi di "
            "bilancio, liquidità, indagini "
            "oppure rischio di delisting."
        )

    elif (
        catalyst_risk is not None
        and catalyst_risk >= 60
    ):
        risks.append(
            "Il rischio legato all'evento "
            "scatenante è elevato "
            f"({catalyst_risk:.0f}/100)."
        )

    quality = row.get(
        "quality_score"
    )

    if (
        quality is not None
        and quality < 45
    ):
        risks.append(
            "I fondamentali disponibili "
            "risultano deboli "
            f"(qualità stimata "
            f"{quality:.0f}/100). "
            "Il ribasso potrebbe quindi "
            "essere almeno in parte giustificato."
        )

    liabilities = row.get(
        "liabilities_to_assets"
    )

    if (
        liabilities is not None
        and liabilities >= 0.7
    ):
        risks.append(
            "Il livello delle passività "
            "rispetto agli attivi è elevato. "
            "Questo aumenta il rischio in caso "
            "di ulteriore peggioramento."
        )

    revenue_growth = row.get(
        "revenue_growth_pct"
    )

    if (
        revenue_growth is not None
        and revenue_growth < 0
    ):
        risks.append(
            "La crescita dei ricavi risulta "
            f"negativa ({revenue_growth:.1f}%). "
            "Il dato indica un possibile "
            "deterioramento del business."
        )

    eps_growth = row.get(
        "eps_growth_pct"
    )

    if (
        eps_growth is not None
        and eps_growth < 0
    ):
        risks.append(
            "La crescita degli utili risulta "
            f"negativa ({eps_growth:.1f}%)."
        )

    fcf_margin = row.get(
        "fcf_margin_pct"
    )

    if (
        fcf_margin is not None
        and fcf_margin < 0
    ):
        risks.append(
            "Il flusso di cassa libero è "
            "negativo rispetto ai ricavi. "
            "L'azienda sta assorbendo cassa."
        )

    if row.get("fundamentals_error"):
        risks.append(
            "Non è stato possibile recuperare "
            "tutti i dati fondamentali dal "
            "provider. L'analisi deve essere "
            "considerata incompleta."
        )

    if row.get("error"):
        risks.append(
            "Sono presenti dati incompleti "
            "oppure un errore nell'analisi "
            "di questo titolo."
        )

    if not risks:
        risks.append(
            "Dai dati disponibili non emergono "
            "rischi strutturali evidenti. "
            "Questo non esclude eventi o rischi "
            "che il modello non può rilevare."
        )

    return risks


def build_ticker_narrative(
    row: dict,
    thresholds: dict | None = None,
) -> dict:

    classification = (
        classify_opportunity(
            row.get(
                "opportunity_score",
                0,
            ),
            thresholds,
        )
    )

    return {
        "classification": classification,
        "why_anomaly": (
            why_it_might_be_anomaly(
                row
            )
        ),
        "why_not": (
            why_it_might_not_be(
                row
            )
        ),
    }
