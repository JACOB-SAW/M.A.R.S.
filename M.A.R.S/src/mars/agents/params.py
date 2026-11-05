from dataclasses import dataclass, asdict, fields


@dataclass
class AgentParams:
    """The tunable 'personality' of an agent. All values are floats."""
    risk_tolerance: float = 0.5   # willingness to approach danger
    exploration: float = 0.1      # chance of acting randomly
    horizon: float = 3.0          # how far ahead it senses danger
    safety: float = 0.5           # how much it values avoiding hazards
    novelty: float = 0.3          # how much it values unvisited cells
    efficiency: float = 0.7       # how much it values heading to the goal

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AgentParams":
        names = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in d.items() if k in names})

    @classmethod
    def names(cls) -> list:
        return [f.name for f in fields(cls)]
