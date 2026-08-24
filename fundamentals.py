from __future__ import annotations

import math


def safe(value):
    try:
        number = float(value)

        if (
            math.isnan(number)
            or math.isinf(number)
        ):
            return None

        return number

    except Exception:
        return None


def clamp(
    value,
    minimum=0.0,
    maximum=100.0,
):
    return max(
        minimum,
        min(
            maximum,
            float(value),
        ),
    )


def _score_range(
    value,
    excellent,
    good,
    neutral,
    bad,
    higher_is_better=True,
):
    value = safe(value)

    if value is None:
        return None

    if higher_is_better:
        if value >= excellent:
            return 100.0

        if value >= good:
            return 80.0

        if value >= neutral:
            return 60.0

        if value >= bad:
            return 40.0

        return 15.0

    if value <= excellent:
        return 100.0

    if value <= good:
        return 80.0

    if value <= neutral:
        return 60.0

    if value <= bad:
        return 40.0

    return 15.0


def valuation_score(
    metrics: dict,
) -> float:
    """
    0 = valutazione molto elevata.
    100 = valutazione relativamente contenuta.

    Combina più indicatori quando disponibili.
    Non rappresenta un consiglio finanziario.
    """

    scores = []
    weights = []

    pe = safe(
        metrics.get("pe_ratio")
    )

    forward_pe = safe(
        metrics.get("forward_pe")
    )

    ev_ebitda = safe(
        metrics.get("ev_to_ebitda")
    )

    ev_sales = safe(
        metrics.get("ev_to_sales")
    )

    price_to_sales = safe(
        metrics.get("price_to_sales")
    )

    price_to_book = safe(
        metrics.get("price_to_book")
    )

    peg = safe(
        metrics.get("peg_ratio")
    )

    fcf_yield = safe(
        metrics.get("fcf_yield_pct")
    )

    historical_pe = safe(
        metrics.get(
            "historical_pe_median"
        )
    )

    peer_pe = safe(
        metrics.get("peer_pe_median")
    )

    def add(score, weight):
        if score is not None:
            scores.append(score)
            weights.append(weight)

    if pe is not None and pe > 0:
        add(
            _score_range(
                pe,
                excellent=12,
                good=18,
                neutral=28,
                bad=45,
                higher_is_better=False,
            ),
            1.2,
        )

    if (
        forward_pe is not None
        and forward_pe > 0
    ):
        add(
            _score_range(
                forward_pe,
                excellent=12,
                good=18,
                neutral=28,
                bad=45,
                higher_is_better=False,
            ),
            1.4,
        )

    if (
        ev_ebitda is not None
        and ev_ebitda > 0
    ):
        add(
            _score_range(
                ev_ebitda,
                excellent=8,
                good=12,
                neutral=18,
                bad=28,
                higher_is_better=False,
            ),
            1.2,
        )

    if (
        ev_sales is not None
        and ev_sales > 0
    ):
        add(
            _score_range(
                ev_sales,
                excellent=2,
                good=4,
                neutral=8,
                bad=15,
                higher_is_better=False,
            ),
            0.8,
        )

    if (
        price_to_sales is not None
        and price_to_sales > 0
    ):
        add(
            _score_range(
                price_to_sales,
                excellent=2,
                good=4,
                neutral=8,
                bad=15,
                higher_is_better=False,
            ),
            0.8,
        )

    if (
        price_to_book is not None
        and price_to_book > 0
    ):
        add(
            _score_range(
                price_to_book,
                excellent=1.2,
                good=2.0,
                neutral=4.0,
                bad=8.0,
                higher_is_better=False,
            ),
            0.5,
        )

    if peg is not None and peg > 0:
        add(
            _score_range(
                peg,
                excellent=1.0,
                good=1.5,
                neutral=2.5,
                bad=4.0,
                higher_is_better=False,
            ),
            0.8,
        )

    if fcf_yield is not None:
        add(
            _score_range(
                fcf_yield,
                excellent=8,
                good=5,
                neutral=3,
                bad=0,
                higher_is_better=True,
            ),
            1.3,
        )

    # Confronto con la valutazione storica.
    if (
        pe is not None
        and historical_pe is not None
        and pe > 0
        and historical_pe > 0
    ):
        discount = (
            historical_pe
            / pe
            - 1
        ) * 100

        add(
            _score_range(
                discount,
                excellent=30,
                good=15,
                neutral=0,
                bad=-25,
                higher_is_better=True,
            ),
            1.1,
        )

    # Confronto con i concorrenti.
    if (
        pe is not None
        and peer_pe is not None
        and pe > 0
        and peer_pe > 0
    ):
        peer_discount = (
            peer_pe
            / pe
            - 1
        ) * 100

        add(
            _score_range(
                peer_discount,
                excellent=25,
                good=10,
                neutral=0,
                bad=-20,
                higher_is_better=True,
            ),
            1.0,
        )

    if not scores:
        # Valore neutro usato solamente
        # quando non esiste alcun multiplo.
        return 50.0

    return round(
        sum(
            score * weight
            for score, weight
            in zip(
                scores,
                weights,
            )
        )
        / sum(weights),
        1,
    )


def quality_from_metrics(
    metrics: dict,
) -> float:
    """
    Qualità generale del business.

    0 = qualità molto debole.
    100 = qualità molto elevata.
    """

    score = 50.0

    revenue_growth = safe(
        metrics.get(
            "revenue_growth_pct"
        )
    )

    revenue_growth_3y = safe(
        metrics.get(
            "revenue_growth_3y_pct"
        )
    )

    eps_growth = safe(
        metrics.get("eps_growth_pct")
    )

    gross_margin = safe(
        metrics.get("gross_margin_pct")
    )

    operating_margin = safe(
        metrics.get(
            "operating_margin_pct"
        )
    )

    net_margin = safe(
        metrics.get("net_margin_pct")
    )

    fcf_margin = safe(
        metrics.get("fcf_margin_pct")
    )

    roe = safe(
        metrics.get("roe_pct")
    )

    roic = safe(
        metrics.get("roic_pct")
    )

    if revenue_growth is not None:
        score += max(
            -18,
            min(
                18,
                revenue_growth * 0.55,
            ),
        )

    if revenue_growth_3y is not None:
        score += max(
            -10,
            min(
                10,
                revenue_growth_3y * 0.25,
            ),
        )

    if eps_growth is not None:
        score += max(
            -12,
            min(
                12,
                eps_growth * 0.25,
            ),
        )

    if gross_margin is not None:
        if gross_margin >= 60:
            score += 8

        elif gross_margin >= 40:
            score += 5

        elif gross_margin < 20:
            score -= 6

    if operating_margin is not None:
        if operating_margin >= 20:
            score += 10

        elif operating_margin >= 10:
            score += 6

        elif operating_margin < 0:
            score -= 12

    if net_margin is not None:
        if net_margin >= 15:
            score += 10

        elif net_margin >= 5:
            score += 6

        elif net_margin > 0:
            score += 2

        else:
            score -= 14

    if fcf_margin is not None:
        if fcf_margin >= 15:
            score += 12

        elif fcf_margin >= 5:
            score += 7

        elif fcf_margin < 0:
            score -= 14

    if roe is not None:
        if roe >= 20:
            score += 6

        elif roe >= 10:
            score += 3

        elif roe < 0:
            score -= 5

    if roic is not None:
        if roic >= 15:
            score += 8

        elif roic >= 8:
            score += 4

        elif roic < 0:
            score -= 6

    return round(
        clamp(score),
        1,
    )


def financial_risk_score(
    metrics: dict,
) -> float:
    """
    0 = rischio finanziario basso.
    100 = rischio finanziario elevato.
    """

    risk = 25.0

    debt_to_ebitda = safe(
        metrics.get("debt_to_ebitda")
    )

    net_debt_to_ebitda = safe(
        metrics.get(
            "net_debt_to_ebitda"
        )
    )

    liabilities_to_assets = safe(
        metrics.get(
            "liabilities_to_assets"
        )
    )

    current_ratio = safe(
        metrics.get("current_ratio")
    )

    interest_coverage = safe(
        metrics.get(
            "interest_coverage"
        )
    )

    cash_runway_months = safe(
        metrics.get(
            "cash_runway_months"
        )
    )

    fcf_margin = safe(
        metrics.get("fcf_margin_pct")
    )

    if debt_to_ebitda is not None:
        if debt_to_ebitda >= 6:
            risk += 28

        elif debt_to_ebitda >= 4:
            risk += 18

        elif debt_to_ebitda >= 3:
            risk += 10

        elif debt_to_ebitda <= 1:
            risk -= 8

    if net_debt_to_ebitda is not None:
        if net_debt_to_ebitda >= 5:
            risk += 18

        elif net_debt_to_ebitda >= 3:
            risk += 10

        elif net_debt_to_ebitda <= 0:
            risk -= 8

    if liabilities_to_assets is not None:
        if liabilities_to_assets >= 0.90:
            risk += 18

        elif liabilities_to_assets >= 0.80:
            risk += 10

        elif liabilities_to_assets <= 0.50:
            risk -= 6

    if current_ratio is not None:
        if current_ratio < 0.75:
            risk += 18

        elif current_ratio < 1.0:
            risk += 10

        elif current_ratio >= 2:
            risk -= 6

    if interest_coverage is not None:
        if interest_coverage < 1:
            risk += 25

        elif interest_coverage < 2:
            risk += 15

        elif interest_coverage < 4:
            risk += 7

        elif interest_coverage >= 8:
            risk -= 6

    if cash_runway_months is not None:
        if cash_runway_months < 6:
            risk += 30

        elif cash_runway_months < 12:
            risk += 18

        elif cash_runway_months < 24:
            risk += 8

    if (
        fcf_margin is not None
        and fcf_margin < 0
    ):
        risk += 10

    return round(
        clamp(risk),
        1,
    )


def dilution_risk_score(
    metrics: dict,
) -> float:
    """
    Rischio di emissione di nuove azioni
    o diluizione degli azionisti.
    """

    risk = 15.0

    share_growth = safe(
        metrics.get(
            "shares_outstanding_growth_pct"
        )
    )

    cash_runway = safe(
        metrics.get(
            "cash_runway_months"
        )
    )

    fcf_margin = safe(
        metrics.get("fcf_margin_pct")
    )

    if share_growth is not None:
        if share_growth >= 20:
            risk += 45

        elif share_growth >= 10:
            risk += 30

        elif share_growth >= 5:
            risk += 18

        elif share_growth <= 0:
            risk -= 5

    if (
        fcf_margin is not None
        and fcf_margin < 0
    ):
        risk += 12

    if (
        cash_runway is not None
        and cash_runway < 12
    ):
        risk += 20

    return round(
        clamp(risk),
        1,
    )


def distress_risk_score(
    metrics: dict,
) -> float:
    """
    Stima prudente del rischio di difficoltà
    finanziaria.

    Non rappresenta una previsione certa
    di fallimento.
    """

    risk = financial_risk_score(
        metrics
    )

    market_cap = safe(
        metrics.get("market_cap")
    )

    revenue_growth = safe(
        metrics.get(
            "revenue_growth_pct"
        )
    )

    net_margin = safe(
        metrics.get("net_margin_pct")
    )

    fcf_margin = safe(
        metrics.get("fcf_margin_pct")
    )

    dilution = dilution_risk_score(
        metrics
    )

    if market_cap is not None:
        if market_cap < 500_000_000:
            risk += 18

        elif market_cap < 2_000_000_000:
            risk += 8

    if (
        revenue_growth is not None
        and revenue_growth <= -20
    ):
        risk += 12

    if (
        net_margin is not None
        and net_margin < -15
    ):
        risk += 12

    if (
        fcf_margin is not None
        and fcf_margin < -15
    ):
        risk += 15

    risk += dilution * 0.20

    return round(
        clamp(risk),
        1,
    )


def value_trap_risk(
    metrics: dict,
    technical: dict,
) -> float:
    """
    Rischio che il titolo sembri economico
    solamente perché il business sta
    realmente peggiorando.
    """

    risk = 30.0

    quality = safe(
        metrics.get("quality_score")
    )

    valuation = safe(
        metrics.get("valuation_score")
    )

    revenue_growth = safe(
        metrics.get(
            "revenue_growth_pct"
        )
    )

    eps_growth = safe(
        metrics.get("eps_growth_pct")
    )

    net_margin = safe(
        metrics.get("net_margin_pct")
    )

    fcf_margin = safe(
        metrics.get("fcf_margin_pct")
    )

    guidance_change = safe(
        metrics.get(
            "guidance_change_pct"
        )
    )

    analyst_revision = safe(
        metrics.get(
            "analyst_eps_revision_pct"
        )
    )

    financial_risk = safe(
        metrics.get(
            "financial_risk_score"
        )
    )

    distress_risk = safe(
        metrics.get(
            "distress_risk_score"
        )
    )

    dilution_risk = safe(
        metrics.get(
            "dilution_risk_score"
        )
    )

    if quality is not None:
        risk += (
            50 - quality
        ) * 0.35

    # Se il titolo è ancora caro dopo il crollo,
    # il rischio aumenta.
    if valuation is not None:
        risk += (
            50 - valuation
        ) * 0.25

    if (
        revenue_growth is not None
        and revenue_growth < 0
    ):
        risk += min(
            15,
            abs(
                revenue_growth
            )
            * 0.35,
        )

    if (
        eps_growth is not None
        and eps_growth < 0
    ):
        risk += min(
            12,
            abs(
                eps_growth
            )
            * 0.20,
        )

    if (
        net_margin is not None
        and net_margin < 0
    ):
        risk += 10

    if (
        fcf_margin is not None
        and fcf_margin < 0
    ):
        risk += 12

    if (
        guidance_change is not None
        and guidance_change < 0
    ):
        risk += min(
            18,
            abs(
                guidance_change
            )
            * 0.35,
        )

    if (
        analyst_revision is not None
        and analyst_revision < 0
    ):
        risk += min(
            12,
            abs(
                analyst_revision
            )
            * 0.25,
        )

    if financial_risk is not None:
        risk += (
            financial_risk
            - 50
        ) * 0.20

    if distress_risk is not None:
        risk += (
            distress_risk
            - 40
        ) * 0.25

    if dilution_risk is not None:
        risk += (
            dilution_risk
            - 30
        ) * 0.15

    return_60d = safe(
        technical.get(
            "return_60d_pct"
        )
    )

    if (
        return_60d is not None
        and return_60d <= -30
        and (
            quality
            if quality is not None
            else 50
        )
        < 50
    ):
        risk += 10

    return round(
        clamp(risk),
        1,
    )


def confidence_score(
    metrics: dict,
) -> float:
    """
    Misura la completezza dell'analisi.

    Se mancano dati importanti, il punteggio
    rimane basso e Market Anomaly deve
    dichiararlo chiaramente.
    """

    important_fields = [
        "revenue_growth_pct",
        "eps_growth_pct",
        "net_margin_pct",
        "fcf_margin_pct",
        "pe_ratio",
        "forward_pe",
        "ev_to_ebitda",
        "debt_to_ebitda",
        "current_ratio",
        "market_cap",
        "shares_outstanding_growth_pct",
    ]

    available = sum(
        1
        for field in important_fields
        if safe(
            metrics.get(field)
        )
        is not None
    )

    completeness = (
        available
        / len(important_fields)
        * 100
    )

    return round(
        clamp(completeness),
        1,
    )


def enrich_fundamental_scores(
    metrics: dict,
) -> dict:
    """
    Normalizza i nomi delle metriche,
    calcola tutti i punteggi fondamentali
    e li aggiunge al risultato.
    """

    result = dict(metrics)

    # SEC utilizza inizialmente i nomi
    # approx_pe e approx_ps.
    #
    # Li convertiamo nei nomi usati
    # dal motore di valutazione.
    if safe(
        result.get("pe_ratio")
    ) is None:
        result["pe_ratio"] = safe(
            result.get("approx_pe")
        )

    if safe(
        result.get("price_to_sales")
    ) is None:
        result[
            "price_to_sales"
        ] = safe(
            result.get("approx_ps")
        )

    quality = quality_from_metrics(
        result
    )

    valuation = valuation_score(
        result
    )

    financial_risk = (
        financial_risk_score(
            result
        )
    )

    dilution_risk = (
        dilution_risk_score(
            result
        )
    )

    distress_risk = (
        distress_risk_score(
            result
        )
    )

    confidence = confidence_score(
        result
    )

    result.update({
        "quality_score": quality,
        "valuation_score": valuation,
        "financial_risk_score": (
            financial_risk
        ),
        "dilution_risk_score": (
            dilution_risk
        ),
        "distress_risk_score": (
            distress_risk
        ),
        "confidence_score": confidence,
    })

    return result
