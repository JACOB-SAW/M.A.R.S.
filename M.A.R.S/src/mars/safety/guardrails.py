from mars.agents.params import AgentParams
from mars.evolution.bounds import apply_bounds


class Guardrails:
    """Enforces protected.yaml. Everything that changes the system goes
    through here; the system itself never edits the config."""

    def __init__(self, protected: dict):
        self.p = protected

    def clamp_params(self, new: AgentParams,
                     old: AgentParams | None = None) -> AgentParams:
        return apply_bounds(new, old, self.p["param_bounds"],
                            self.p["max_param_delta"])

    @property
    def min_population(self) -> int:
        return self.p["population"]["min"]

    @property
    def max_population(self) -> int:
        return self.p["population"]["max"]

    def cap_steps(self, requested: int) -> int:
        return min(requested, self.p["max_steps_per_cycle"])

    def cap_cycles(self, requested: int) -> int:
        return min(requested, self.p["max_cycles_per_run"])

    @property
    def purpose_needs_approval(self) -> bool:
        return bool(self.p["require_approval_for_purpose_change"])

    def check_population(self, n: int) -> None:
        if not (self.min_population <= n <= self.max_population):
            raise RuntimeError(
                f"Population size {n} outside protected limits "
                f"[{self.min_population}, {self.max_population}]")
