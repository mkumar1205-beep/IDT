def quality_score(pothole_count: int, damage_area_pct: float) -> tuple[int, str, str]:
    """
    Calculate road quality score.

    Uses two independent signals and takes the worse of the two:
      - Count penalty:  each pothole subtracts heavily
      - Area penalty:   damage spread is weighted strongly

    This prevents a road with many potholes OR large damage
    from being rated Good.
    """
    # Penalty from pothole count (each pothole = -8 points)
    count_score = max(0, 100 - (pothole_count * 8))

    # Penalty from damage area (each % of damage = -1.5 points)
    area_score = max(0, 100 - (damage_area_pct * 1.5))

    # Take the WORSE of the two signals
    score = int(min(count_score, area_score))

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