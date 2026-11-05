def rank(scores: dict) -> list:
    """Agent ids, best first."""
    return sorted(scores, key=lambda aid: -scores[aid])


def split(ranked: list, elite_fraction: float, n_replace: int):
    """Returns (elites, middle, worst)."""
    n = len(ranked)
    n_elite = max(1, round(n * elite_fraction))
    n_replace = min(n_replace, n - n_elite)
    elites = ranked[:n_elite]
    worst = ranked[n - n_replace:] if n_replace > 0 else []
    middle = ranked[n_elite:n - n_replace] if n_replace > 0 else ranked[n_elite:]
    return elites, middle, worst
