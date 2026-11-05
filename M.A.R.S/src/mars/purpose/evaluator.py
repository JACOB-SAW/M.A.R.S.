from mars.purpose.anchor import anchor_score


class PurposeEvaluator:
    """Collects per-cycle metrics for each purpose and scores them
    with the fixed anchor criteria."""

    def __init__(self):
        self.records: dict = {}

    def record(self, purpose_name: str, metrics: dict) -> None:
        self.records.setdefault(purpose_name, []).append(metrics)

    def score(self, purpose_name: str, window: int):
        recent = self.records.get(purpose_name, [])[-window:]
        if len(recent) < 2:
            return None
        return anchor_score(recent)

    def best(self, names: list, window: int):
        scored = [(self.score(n, window), n) for n in names]
        scored = [(s, n) for s, n in scored if s is not None]
        if not scored:
            return names[0], {}
        scored.sort(reverse=True)
        return scored[0][1], {n: round(s, 4) for s, n in scored}
