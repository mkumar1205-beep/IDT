from typing import Optional


def quality_score(
    pothole_count: int,
    damage_area_pct: float,
    severity_counts: Optional[dict] = None,
) -> tuple[int, str, str]:
    """
    Calculate road quality score.

    Uses count, area, and severity so a road with confirmed potholes
    cannot still be rated Good just because the damaged area is small.
    """
    severity_counts = severity_counts or {}
    small = severity_counts.get("small", 0)
    medium = severity_counts.get("medium", 0)
    large = severity_counts.get("large", 0)

    if not severity_counts and pothole_count:
        small = pothole_count

    severity_penalty = (small * 12) + (medium * 22) + (large * 35)
    area_penalty = damage_area_pct * 2.5
    score = max(0, int(100 - severity_penalty - area_penalty))

    if pothole_count > 0:
        score = min(score, 74)
    if medium > 0 or pothole_count >= 2 or damage_area_pct >= 2:
        score = min(score, 69)
    if large > 0 or pothole_count >= 4 or damage_area_pct >= 6:
        score = min(score, 44)

    if score >= 75:
        return score, "Good", "#1D9E75"
    elif score >= 45:
        return score, "Moderate", "#BA7517"
    else:
        return score, "Poor", "#A32D2D"


def severity_summary(severity_counts: dict) -> str:
    s = severity_counts.get("small", 0)
    m = severity_counts.get("medium", 0)
    l = severity_counts.get("large", 0)
    parts = []
    if s: parts.append(f"{s} small")
    if m: parts.append(f"{m} medium")
    if l: parts.append(f"{l} large")
    return ", ".join(parts) if parts else "none"
