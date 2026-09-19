def pairwise_preference_judge(comparisons, criteria_weights, tie_threshold):
    if not comparisons:
        return {
            "results": [],
            "win_rate_a": 0.0,
            "win_rate_b": 0.0,
            "tie_rate": 0.0,
            "avg_margin": 0.0
        }

    # 1. Normalize weights
    total_weight = sum(criteria_weights.values())
    normalized_weights = {
        k: v / total_weight
        for k, v in criteria_weights.items()
    }

    results = []
    wins_a = 0
    wins_b = 0
    ties = 0
    total_margin = 0.0

    # 2. Evaluate each comparison
    for comp in comparisons:
        score_a = sum(
            comp["scores_a"][criterion] * weight
            for criterion, weight in normalized_weights.items()
        )

        score_b = sum(
            comp["scores_b"][criterion] * weight
            for criterion, weight in normalized_weights.items()
        )

        diff = score_a - score_b
        margin = abs(diff)

        if margin <= tie_threshold:
            winner = "tie"
            ties += 1
        elif diff > 0:
            winner = "A"
            wins_a += 1
        else:
            winner = "B"
            wins_b += 1

        total_margin += margin

        results.append({
            "id": comp["id"],
            "winner": winner,
            "margin": round(margin, 4)
        })

    # 3. Aggregate statistics
    n = len(comparisons)

    return {
        "results": results,
        "win_rate_a": round(wins_a / n, 4),
        "win_rate_b": round(wins_b / n, 4),
        "tie_rate": round(ties / n, 4),
        "avg_margin": round(total_margin / n, 4)
    }