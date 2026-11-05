class Arbiter:
    """Weighted vote: each agent's vote counts weight * confidence."""

    def decide(self, proposals, weights: dict, rng):
        tally: dict = {}
        for p in proposals:
            tally[p.action] = tally.get(p.action, 0.0) + weights[p.agent_id] * p.confidence
        best = max(tally.values())
        winners = sorted(a for a, v in tally.items() if abs(v - best) < 1e-9)
        return rng.choice(winners), tally
