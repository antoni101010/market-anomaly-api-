
import math

def safe(v):
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except Exception:
        return None

def quality_from_metrics(m):
    score = 50.0
    rg = safe(m.get("revenue_growth_pct"))
    nm = safe(m.get("net_margin_pct"))
    la = safe(m.get("liabilities_to_assets"))
    fm = safe(m.get("fcf_margin_pct"))

    if rg is not None:
        score += max(-18, min(18, rg * 0.65))
    if nm is not None:
        if nm > 15: score += 14
        elif nm > 5: score += 8
        elif nm > 0: score += 3
        else: score -= 16
    if la is not None:
        if la < 0.50: score += 12
        elif la < 0.70: score += 5
        elif la > 0.90: score -= 15
        elif la > 0.80: score -= 8
    if fm is not None:
        if fm > 15: score += 12
        elif fm > 5: score += 6
        elif fm < 0: score -= 12

    return max(0.0, min(100.0, score))

def value_trap_risk(m, technical):
    risk = 35.0
    q = safe(m.get("quality_score"))
    rg = safe(m.get("revenue_growth_pct"))
    nm = safe(m.get("net_margin_pct"))
    fm = safe(m.get("fcf_margin_pct"))
    la = safe(m.get("liabilities_to_assets"))

    if q is not None:
        risk += max(-20, min(25, (55 - q) * 0.55))
    if rg is not None and rg < 0:
        risk += min(18, abs(rg) * 0.6)
    if nm is not None and nm < 0:
        risk += 15
    if fm is not None and fm < 0:
        risk += 12
    if la is not None and la > 0.85:
        risk += 12

    # Crollo violento + qualità debole = maggiore rischio strutturale.
    if technical.get("return_60d_pct", 0) < -30 and (q or 50) < 55:
        risk += 10

    return max(0.0, min(100.0, risk))
