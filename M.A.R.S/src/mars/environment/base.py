from abc import ABC, abstractmethod


class Environment(ABC):
    actions: dict  # name -> (dx, dy)

    @abstractmethod
    def reset(self) -> dict: ...

    @abstractmethod
    def state(self) -> dict: ...

    @abstractmethod
    def step(self, action: str):
        """Returns (state, reward, done, info)."""
