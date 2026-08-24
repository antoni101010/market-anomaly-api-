import re
from datetime import datetime, timezone


STRUCTURAL_NEGATIVE = {
    "fraud": 35,
    "accounting": 30,
    "restatement": 35,
    "material weakness": 25,
    "bankruptcy": 45,
    "default": 40,
    "liquidity": 30,
    "covenant": 30,
    "subpoena": 30,
    "investigation": 28,
    "criminal": 40,
    "sec investigation": 38,
    "antitrust": 22,
    "regulatory": 20,
    "lawsuit": 16,
    "litigation": 16,
    "data breach": 22,
    "cybersecurity incident": 22,
    "impairment": 18,
    "going concern": 45,
    "delisting": 40,
    "auditor resignation": 38,
}


TEMPORARY_NEGATIVE = {
    "lowered guidance": 18,
    "cuts guidance": 18,
    "guidance cut": 18,
    "misses estimates": 14,
    "missed estimates": 14,
    "revenue miss": 14,
    "earnings miss": 14,
    "weak demand": 12,
    "inventory": 10,
    "restructuring": 10,
    "one-time charge": 8,
    "foreign exchange": 6,
    "currency headwind": 6,
    "macro": 8,
    "slowdown": 10,
    "delay": 8,
}


POSITIVE = {
    "raises guidance": -18,
    "raised guidance": -18,
    "beats estimates": -14,
    "beat estimates": -14,
    "share repurchase": -12,
    "buyback": -12,
    "authorization": -5,
    "dividend increase": -8,
    "contract award": -10,
    "approval": -10,
    "record revenue": -10,
    "record earnings": -10,
}


EARNINGS_WORDS = (
    "earnings",
    "quarter",
    "quarterly",
    "results",
    "revenue",
    "eps",
    "guidance",
    "outlook",
    "margin",
)


def _blob(items):
    chunks = []

    for item in items or []:
        title = str(
            item.get("title")
            or item.get("headline")
            or ""
        )

        text = str(
            item.get("text")
            or item.get("summary")
            or item.get("description")
            or ""
        )

        chunks.append(
            (title + " " + text).lower()
        )

    return "\n".join(chunks)


def classify_catalysts(
    items,
    filings=None,
):
    text = _blob(items)

    hits_struct = []
    hits_temp = []
    hits_pos = []

    risk = 45.0

    for phrase, weight in STRUCTURAL_NEGATIVE.items():
        if phrase in text:
            hits_struct.append(phrase)
            risk += weight

    for phrase, weight in TEMPORARY_NEGATIVE.items():
        if phrase in text:
            hits_temp.append(phrase)
            risk += weight

    for phrase, weight in POSITIVE.items():
        if phrase in text:
            hits_pos.append(phrase)
            risk += weight

    earnings_related = any(
        word in text
        for word in EARNINGS_WORDS
    )

    filing_forms = [
        str(item.get("form", ""))
        for item in (filings or [])
    ]

    if "8-K" in filing_forms and not text:
        risk += 4

    if any(
        form in filing_forms
        for form in ("10-Q", "10-K")
    ):
        earnings_related = True

    if hits_struct:
        label = (
            "Possibile rischio strutturale"
        )

    elif hits_temp and not hits_pos:
        label = (
            "Catalizzatore negativo "
            "potenzialmente temporaneo"
        )

    elif hits_pos and not hits_temp:
        label = (
            "Catalizzatore prevalentemente "
            "positivo"
        )

    elif hits_temp and hits_pos:
        label = "Catalizzatori misti"

    elif items:
        label = "Causa non classificata"

    elif filings:
        label = "Solo filing SEC rilevati"

    else:
        label = (
            "Nessun catalizzatore disponibile"
        )

    risk = max(
        0.0,
        min(
            100.0,
            risk,
        ),
    )

    explanations = []

    if hits_struct:
        explanations.append(
            "rischi strutturali: "
            + ", ".join(
                sorted(
                    set(hits_struct)
                )[:5]
            )
        )

    if hits_temp:
        explanations.append(
            "fattori temporanei/operativi: "
            + ", ".join(
                sorted(
                    set(hits_temp)
                )[:5]
            )
        )

    if hits_pos:
        explanations.append(
            "fattori positivi: "
            + ", ".join(
                sorted(
                    set(hits_pos)
                )[:5]
            )
        )

    if earnings_related:
        explanations.append(
            "contenuto collegato a "
            "risultati/guidance"
        )

    return {
        "catalyst_label": label,
        "catalyst_risk": round(
            risk,
            1,
        ),
        "earnings_related": bool(
            earnings_related
        ),
        "structural_hits": ", ".join(
            sorted(
                set(hits_struct)
            )
        ),
        "temporary_hits": ", ".join(
            sorted(
                set(hits_temp)
            )
        ),
        "positive_hits": ", ".join(
            sorted(
                set(hits_pos)
            )
        ),
        "catalyst_explanation": (
            "; ".join(explanations)
            if explanations
            else (
                "Nessuna causa forte "
                "identificata automaticamente."
            )
        ),
        "catalyst_items": items or [],
        "recent_filings": filings or [],
    }


def opportunity_score(
    anomaly_score,
    quality_score,
    value_trap_risk,
    catalyst_risk,
    valuation_score=50,
    financial_risk=50,
    distress_risk=50,
    dilution_risk=50,
    confidence_score=100,
):
    """
    Sintesi analitica.

    Non rappresenta una probabilità,
    una previsione di rendimento o un
    consiglio di investimento.

    Un forte ribasso da solo non basta.

    Una valutazione ancora elevata,
    la fragilità finanziaria e i dati
    incompleti riducono esplicitamente
    il punteggio.
    """

    score = (
        float(anomaly_score) * 0.28
        + float(quality_score) * 0.12
        + float(valuation_score) * 0.17
        + (
            100
            - float(value_trap_risk)
        )
        * 0.14
        + (
            100
            - float(catalyst_risk)
        )
        * 0.10
        + (
            100
            - float(financial_risk)
        )
        * 0.07
        + (
            100
            - float(distress_risk)
        )
        * 0.06
        + (
            100
            - float(dilution_risk)
        )
        * 0.03
    )

    # Confidence non aumenta artificialmente
    # il punteggio.
    #
    # Quando mancano dati importanti,
    # limita quanto il motore può dichiararsi
    # convinto dell'anomalia.
    confidence = max(
        0.0,
        min(
            100.0,
            float(confidence_score),
        ),
    )

    score *= (
        0.55
        + (
            confidence
            / 100.0
        )
        * 0.45
    )

    return round(
        max(
            0.0,
            min(
                100.0,
                score,
            ),
        ),
        1,
    )
