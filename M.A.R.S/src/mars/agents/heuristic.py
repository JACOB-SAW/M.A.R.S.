import math

from mars.agents.base import Agent, Proposal


class HeuristicAgent(Agent):
    """A lightweight rule-based agent whose behaviour is set by its params."""

    def _score_cell(self, pos, state) -> float:
        p = self.params
        gx, gy = state["goal"]
        dist = abs(pos[0] - gx) + abs(pos[1] - gy)
        visits = state["visits"].get(pos, 0)

        radius = math.ceil(p.horizon / 2)
        danger = 0.0
        for hx, hy in state["hazards"]:
            d = abs(pos[0] - hx) + abs(pos[1] - hy)
            if d == 0:
                danger += 10.0
            elif d <= radius:
                danger += 1.0 / d

        caution = p.safety * (1.2 - p.risk_tolerance)
        return (-p.efficiency * dist
                + 1.5 * p.novelty / (1 + visits)
                - caution * danger)

    def propose(self, state, actions, rng) -> Proposal:
        if rng.random() < self.params.exploration:
            return Proposal(self.id, rng.choice(sorted(actions)), 0.3)

        size = state["size"]
        x, y = state["pos"]
        scored = []
        for name, (dx, dy) in actions.items():
            nx = min(max(x + dx, 0), size - 1)
            ny = min(max(y + dy, 0), size - 1)
            scored.append((self._score_cell((nx, ny), state), name))

        scored.sort(key=lambda s: -s[0])
        best = scored[0][0]
        tied = [n for s, n in scored if abs(s - best) < 1e-9]
        action = rng.choice(sorted(tied))
        second = next((s for s, n in scored if n not in tied), best)
        confidence = 0.5 + 0.5 * math.tanh(best - second)
        return Proposal(self.id, action, confidence)
