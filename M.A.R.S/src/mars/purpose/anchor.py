"""ANCHOR: the fixed yardstick used to judge candidate purposes.

These criteria are measured from RAW environment results (not purpose-shaped
reward), so a purpose can't win just by redefining success in its own favour.
Treat this file as part of the protected core: change it only by hand.
"""
import math
from types import MappingProxyType

ANCHOR_WEIGHTS = MappingProxyType({
    "learning_progress": 0.4,   # is raw reward trending upward?
    "consistency": 0.3,         # is performance stable?
    "safety": 0.3,              # does it avoid hazards?
})


def _mean(xs):
    return sum(xs) / len(xs)


def learning_progress(rewards: list) -> float:
    """Least-squares slope of reward over cycles, normalised to (-1, 1)."""
    n = len(rewards)
    if n < 2:
        return 0.0
    xs = range(n)
    mx, my = _mean(list(xs)), _mean(rewards)
    den = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, rewards)) / den
    return math.tanh(slope / (abs(my) + 1.0))


def consistency(rewards: list) -> float:
    if len(rewards) < 2:
        return 0.0
    m = _mean(rewards)
    std = math.sqrt(_mean([(r - m) ** 2 for r in rewards]))
    return 1.0 / (1.0 + std / (abs(m) + 1.0))


def safety(hazard_rates: list) -> float:
    return 1.0 - min(1.0, _mean(hazard_rates) * 10.0)


def anchor_score(records: list) -> float:
    """records: list of per-cycle metric dicts."""
    rewards = [r["raw_reward"] for r in records]
    rates = [r["hazard_rate"] for r in records]
    w = ANCHOR_WEIGHTS
    return (w["learning_progress"] * learning_progress(rewards)
            + w["consistency"] * consistency(rewards)
            + w["safety"] * safety(rates))
