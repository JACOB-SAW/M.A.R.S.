import math
from dataclasses import dataclass, field


@dataclass
class Purpose:
    """A candidate answer to 'what am I for?'.
    The weights define how raw events are turned into reward."""
    name: str
    description: str
    weights: dict = field(default_factory=dict)

    def reward(self, info: dict) -> float:
        w = self.weights
        return (w["goal"] * info["goal"]
                + w["novelty"] * info["new_cell"]
                - w["hazard"] * info["hazard"])

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description,
                "weights": dict(self.weights)}


def default_hypotheses() -> list:
    return [
        Purpose("reach_goals", "Achieve as many goals as possible",
                {"goal": 10.0, "novelty": 0.0, "hazard": 2.0}),
        Purpose("explore", "Discover as much of the world as possible",
                {"goal": 2.0, "novelty": 3.0, "hazard": 1.0}),
        Purpose("stay_safe", "Avoid harm above all else",
                {"goal": 4.0, "novelty": 0.2, "hazard": 10.0}),
        Purpose("balanced", "Balance achievement, discovery and safety",
                {"goal": 6.0, "novelty": 1.0, "hazard": 5.0}),
    ]


def make_variant(base: Purpose, rng, index: int) -> Purpose:
    """Generate a new candidate purpose by perturbing an existing one."""
    weights = {k: min(15.0, max(0.0, v * math.exp(rng.gauss(0, 0.5))))
               for k, v in base.weights.items()}
    return Purpose(f"variant_{index}",
                   f"Variant of {base.name}", weights)
