from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from mars.agents.params import AgentParams


@dataclass
class Proposal:
    agent_id: str
    action: str
    confidence: float  # 0..1


class Agent(ABC):
    """Anything that can look at a state and propose an action.

    Swap in an LLM-backed agent later by subclassing this.
    """

    def __init__(self, agent_id: str, name: str, params: AgentParams,
                 generation: int = 0, parent: Optional[str] = None):
        self.id = agent_id
        self.name = name
        self.params = params
        self.generation = generation
        self.parent = parent

    @abstractmethod
    def propose(self, state: dict, actions: dict, rng) -> Proposal:
        ...

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "generation": self.generation,
            "parent": self.parent, "params": self.params.to_dict(),
        }
