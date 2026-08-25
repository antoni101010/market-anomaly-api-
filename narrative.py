"""Testi sintetici, specifici e prudenti per dashboard e dettaglio titolo."""

from __future__ import annotations

import math


DEFAULT_THRESHOLDS = {"strong": 75, "possible": 60, "review": 45}

FIELD_LABELS = {
    "pe_ratio": "P/E",
    "forward_pe": "P/E forward",
    "price_to_sales": "prezzo/ricavi",
    "price_to_book": "prezzo/patrimonio",
    "ev_to_ebitda": "EV/EBITDA",
    "fcf_yield_pct": "rendimento del free cash flow",
    "revenue_growth_pct": "crescita dei ricavi",
    "net_margin_pct": "margine netto",
    "fcf_margin_pct": "margine del free cash flow",
    "liabilities_to_assets": "passività/attivi",
    "debt_to_ebitda": "debito/EBITDA",
    "current_ratio": "liquidità corrente",
    "interest_coverage": "copertura degli interessi",
    "shares_outstanding_growth_pct": "variazione delle azioni in circolazione",
    "market_cap": "capitalizzazione",
    "cash_runway_months": "autonomia di cassa",
}


def _number(value):
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def _missing_fields(row: dict) -> list[str]:
    raw = row.get("missing_fundamental_fields") or []
    if isinstance(raw, str):
        raw = [item.strip() for item in raw.split(",") if item.strip()]
    status = str(row.get("cash_runway_status") or "")
    cleaned = []
    for item in raw:
        field = str(item)
        if field == "cash_runway_months" and status == "not_applicable_positive_fcf":
            continue
        cleaned.append(FIELD_LABELS.get(field, field.replace("_", " ")))
    return cleaned


def data_gaps(row: dict) -> list[str]:
    """Elenca dati assenti o non affidabili senza usare frasi generiche."""
    gaps = []
    price_status = str(row.get("price_status") or row.get("price_validation") or "")
    if price_status in {"conflict", "provider_conflict"}:
        gaps.append("prezzo recente non coerente con lo storico")
    elif price_status in {"stale", "unknown"}:
        gaps.append("prezzo recente non verificato")

    missing = _missing_fields(row)
    if missing:
        preview = ", ".join(missing[:4])
        if len(missing) > 4:
            preview += f" e altri {len(missing) - 4} dati"
        gaps.append(preview)

    if row.get("catalyst_label") in {
        "Non analizzato",
        "Nessun catalizzatore disponibile",
    }:
        gaps.append("causa del movimento non identificata")
    return gaps


def classify_opportunity(
    score: float | None,
    thresholds: dict | None = None,
    row: dict | None = None,
) -> dict:
    t = thresholds or DEFAULT_THRESHOLDS
    value = _number(score) or 0.0
    row = row or {}
    confidence = _number(row.get("confidence_score"))
    price_status = str(row.get("price_status") or row.get("price_validation") or "")

    if price_status in {"conflict", "provider_conflict"}:
        label, key = "PREZZO DA VERIFICARE", "warning"
    elif confidence is None or confidence < 25:
        label, key = "DATI INSUFFICIENTI", "insufficient"
    elif value >= t["strong"]:
        label, key = "SOMIGLIANZA STORICA ALTA", "strong"
    elif value >= t["possible"]:
        label, key = "SOMIGLIANZA STORICA MODERATA", "possible"
    elif value >= t["review"]:
        label, key = "SOMIGLIANZA STORICA LIMITATA", "review"
    else:
        label, key = "SOMIGLIANZA STORICA BASSA", "low"
    return {"label": label, "key": key, "score": round(value, 1)}


def _sector_context(row: dict) -> str | None:
    sector = " ".join(filter(None, [
        str(row.get("light_sector") or row.get("sector") or ""),
        str(row.get("light_industry") or row.get("industry") or ""),
    ])).lower()
    if any(word in sector for word in ("bank", "financial", "insurance")):
        return "Per banche e assicurazioni il P/E da solo è poco informativo: contano soprattutto patrimonio, qualità del credito e capitale regolamentare."
    if any(word in sector for word in ("reit", "real estate")):
        return "Per i REIT utili contabili e P/E possono distorcere il confronto: FFO, debito e copertura delle distribuzioni sono più indicativi."
    if any(word in sector for word in ("biotech", "biotechnology")):
        return "Per una biotech pre-ricavi il valore dipende soprattutto da cassa disponibile, sperimentazioni e rischio di diluizione."
    if any(word in sector for word in ("software", "saas")):
        return "Per il software ricorrente vanno letti insieme crescita, margini, retention e generazione di cassa; il prezzo/ricavi isolato non basta."
    if any(word in sector for word in ("mining", "metals", "gold", "oil", "energy")):
        return "Per le società legate alle materie prime utili e multipli cambiano con il ciclo: debito e costo di produzione aiutano a distinguere il ciclo dal deterioramento."
    return None


def why_it_might_be_anomaly(row: dict) -> list[str]:
    reasons = []
    dd = _number(row.get("drawdown_52w_pct"))
    rel_market = _number(row.get("relative_60d_vs_spy_pct"))
    rel_sector = _number(row.get("relative_60d_vs_sector_pct"))
    volume = _number(row.get("volume_ratio_20d"))
    rsi = _number(row.get("rsi14"))
    worst = _number(row.get("worst_day_20d_pct"))
    quality = _number(row.get("quality_score"))
    valuation = _number(row.get("valuation_score"))

    if dd is not None and dd <= -15:
        intensity = "molto ampio" if dd <= -35 else "significativo"
        reasons.append(
            f"Il ribasso dal massimo a 52 settimane è {abs(dd):.1f}%: un movimento {intensity} rispetto alla storia recente."
        )
    if rel_market is not None and rel_market <= -8:
        reasons.append(
            f"In 60 sedute ha sottoperformato il benchmark di {abs(rel_market):.1f} punti percentuali: il calo non è spiegato soltanto dal mercato."
        )
    if rel_sector is not None and rel_sector <= -8:
        reasons.append(
            f"Rispetto al riferimento settoriale è indietro di {abs(rel_sector):.1f} punti in 60 sedute, quindi il movimento è anche specifico dell'azienda."
        )
    if volume is not None and volume >= 1.5:
        reasons.append(f"Il volume è {volume:.1f} volte la media a 20 sedute, segnale di partecipazione anomala al movimento.")
    if rsi is not None and rsi <= 32:
        reasons.append(f"L'RSI a 14 sedute è {rsi:.0f}, compatibile con una fase di vendita particolarmente intensa.")
    if worst is not None and worst <= -7:
        reasons.append(f"La peggiore seduta recente ha segnato {worst:.1f}%, evidenziando uno shock concentrato.")
    if quality is not None and quality >= 65:
        reasons.append(f"Sui dati disponibili, la qualità fondamentale è {quality:.0f}/100 e non mostra fragilità evidente nel solo quadro numerico.")
    if valuation is not None and valuation >= 65:
        reasons.append(f"La valutazione relativa ottiene {valuation:.0f}/100 sui multipli effettivamente disponibili.")
    if not reasons:
        reasons.append("Il movimento supera i filtri tecnici minimi, ma non presenta ancora un fattore dominante.")
    return reasons


def why_it_might_not_be(row: dict) -> list[str]:
    risks = []
    trap = _number(row.get("value_trap_risk"))
    valuation = _number(row.get("valuation_score"))
    confidence = _number(row.get("confidence_score"))
    financial = _number(row.get("financial_risk_score"))
    distress = _number(row.get("distress_risk_score"))
    dilution = _number(row.get("dilution_risk_score"))
    pe = _number(row.get("pe_ratio") or row.get("approx_pe"))
    ps = _number(row.get("price_to_sales") or row.get("approx_ps"))
    revenue_growth = _number(row.get("revenue_growth_pct"))

    if trap is not None and trap >= 65:
        risks.append(f"Il rischio value trap è {trap:.0f}/100: il deterioramento osservato potrebbe essere strutturale.")
    if valuation is not None and valuation < 40:
        measures = []
        if pe is not None:
            measures.append(f"P/E {pe:.1f}x")
        if ps is not None:
            measures.append(f"prezzo/ricavi {ps:.1f}x")
        detail = f" ({', '.join(measures)})" if measures else ""
        risks.append(f"La valutazione è debole, {valuation:.0f}/100{detail}: il ribasso non ha ancora prodotto multipli favorevoli nel modello.")
    if financial is not None and financial >= 60:
        risks.append(f"Il rischio finanziario è elevato ({financial:.0f}/100) sui dati di debito e liquidità disponibili.")
    if distress is not None and distress >= 60:
        risks.append(f"Il rischio di deterioramento è elevato ({distress:.0f}/100).")
    if dilution is not None and dilution >= 60:
        risks.append(f"Il rischio di diluizione è elevato ({dilution:.0f}/100).")
    if revenue_growth is not None and revenue_growth < 0:
        risks.append(f"I ricavi risultano in contrazione del {abs(revenue_growth):.1f}%: il mercato potrebbe stare prezzando un problema operativo reale.")

    gaps = data_gaps(row)
    if gaps:
        risks.append("La lettura è parziale perché mancano o vanno verificati: " + "; ".join(gaps) + ".")
    elif confidence is not None and confidence < 55:
        risks.append(f"L'affidabilità è {confidence:.0f}/100: il segnale non supera ancora una soglia di robustezza elevata.")

    catalyst = str(row.get("catalyst_label") or "")
    catalyst_risk = _number(row.get("catalyst_risk"))
    if catalyst == "Possibile rischio strutturale":
        risks.append("Le comunicazioni analizzate contengono indicatori di possibile rischio strutturale; serve una verifica dei documenti originali.")
    elif catalyst_risk is not None and catalyst_risk >= 60:
        risks.append(f"Il rischio associato all'evento è {catalyst_risk:.0f}/100.")

    context = _sector_context(row)
    if context:
        risks.append(context)
    if not risks:
        risks.append("I dati disponibili non evidenziano una criticità dominante; restano possibili eventi non ancora riflessi nelle fonti del modello.")
    return risks


def _summary(row: dict, classification: dict) -> str:
    dd = _number(row.get("drawdown_52w_pct"))
    anomaly = _number(row.get("anomaly_score"))
    valuation = _number(row.get("valuation_score"))
    confidence = _number(row.get("confidence_score"))
    gaps = data_gaps(row)

    movement = f"Ribasso di {abs(dd):.1f}% dal massimo" if dd is not None else "Movimento rilevato"
    if classification["key"] == "warning":
        return f"{movement}, ma la quota recente è incoerente: il prezzo va verificato prima di interpretare i punteggi."
    if classification["key"] == "insufficient":
        missing = gaps[0] if gaps else "dati fondamentali e contestuali"
        return f"{movement}; non classificabile in modo robusto finché non sono disponibili {missing}."
    if gaps and confidence is not None and confidence < 55:
        return f"{movement}; segnale tecnico {anomaly or 0:.0f}/100, ma l’affidabilità statistica resta limitata dai dati mancanti."
    if valuation is not None and valuation < 40:
        return f"{movement}; l'anomalia tecnica è accompagnata da un quadro valutativo relativo debole ({valuation:.0f}/100)."
    if classification["key"] in {"strong", "possible"}:
        return f"{movement}; il quadro corrente presenta una somiglianza statistica significativa con casi storici del dataset."
    return f"{movement}; la somiglianza statistica con i casi storici del dataset rimane contenuta."


def build_ticker_narrative(row: dict, thresholds: dict | None = None) -> dict:
    classification = classify_opportunity(
        row.get("opportunity_score"),
        thresholds,
        row=row,
    )
    return {
        "classification": classification,
        "summary": _summary(row, classification),
        "why_anomaly": why_it_might_be_anomaly(row),
        "why_not": why_it_might_not_be(row),
        "data_gaps": data_gaps(row),
        "sector_context": _sector_context(row),
        "informational_only": True,
    }
