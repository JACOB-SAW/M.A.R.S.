class Reputation:
    """Agents earn influence when the actions they backed paid off.

    Within a cycle, an agent is credited with the (purpose-shaped) reward of
    each step where its proposal matched the council's chosen action.
    At cycle end the average is blended into a running score (EMA), and the
    voting weight is derived from that score.
    """

    def __init__(self, decay: float = 0.8, min_weight: float = 0.1,
                 sensitivity: float = 3.0):
        self.decay = decay
        self.min_weight = min_weight
        self.sensitivity = sensitivity
        self.ema: dict = {}
        self._acc: dict = {}
        self._steps = 0

    def register(self, agent_id: str, init: float = 0.0) -> None:
        self.ema[agent_id] = init
        self._acc[agent_id] = 0.0

    def remove(self, agent_id: str) -> None:
        self.ema.pop(agent_id, None)
        self._acc.pop(agent_id, None)

    def mean(self) -> float:
        return sum(self.ema.values()) / len(self.ema) if self.ema else 0.0

    def credit_step(self, proposals, chosen_action: str, shaped_reward: float) -> None:
        for p in proposals:
            if p.action == chosen_action:
                self._acc[p.agent_id] += shaped_reward
        self._steps += 1

    def end_cycle(self) -> dict:
        """Returns this cycle's per-agent score and updates the EMA."""
        steps = max(self._steps, 1)
        scores = {aid: acc / steps for aid, acc in self._acc.items()}
        for aid, s in scores.items():
            self.ema[aid] = self.decay * self.ema[aid] + (1 - self.decay) * s
            self._acc[aid] = 0.0
        self._steps = 0
        return scores

    def weight(self, agent_id: str) -> float:
        return max(self.min_weight, 1.0 + self.sensitivity * self.ema[agent_id])

    def weights(self) -> dict:
        return {aid: self.weight(aid) for aid in self.ema}

    def to_dict(self) -> dict:
        return {"ema": dict(self.ema)}

    def load(self, data: dict) -> None:
        self.ema = dict(data["ema"])
        self._acc = {aid: 0.0 for aid in self.ema}
        self._steps = 0
