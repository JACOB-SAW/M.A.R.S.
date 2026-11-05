from mars.agents.heuristic import HeuristicAgent
from mars.agents.params import AgentParams
from mars.evolution.mutation import mutate
from mars.evolution.selection import rank, split


class Population:
    """Holds the agents and evolves them: elites stay, the worst are
    replaced by mutated clones of elites, the middle get small tunings.
    Every change is clamped by the guardrails."""

    def __init__(self, guard, rng, elite_fraction=0.34,
                 mutation_rate=0.5, mutation_scale=0.15):
        self.guard = guard
        self.rng = rng
        self.elite_fraction = elite_fraction
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.agents: dict = {}
        self._counter = 0

    # ----- creation -----
    def spawn(self, name, params, generation=0, parent=None):
        self._counter += 1
        agent = HeuristicAgent(f"a{self._counter}", name, params, generation, parent)
        self.agents[agent.id] = agent
        return agent

    def add_from_templates(self, templates: list) -> None:
        for t in templates:
            t = dict(t)
            name = t.pop("name")
            params = self.guard.clamp_params(AgentParams.from_dict(t))
            self.spawn(name, params)
        self.guard.check_population(len(self.agents))

    # ----- evolution -----
    def evolve(self, scores: dict, reputation) -> list:
        events = []
        bounds = self.guard.p["param_bounds"]
        ranked = rank({aid: scores[aid] for aid in self.agents})
        n_replace = max(1, len(ranked) // 3)
        elites, middle, worst = split(ranked, self.elite_fraction, n_replace)

        # tune the middle (small changes)
        for aid in middle:
            agent = self.agents[aid]
            proposed = mutate(agent.params, self.rng, self.mutation_rate,
                              self.mutation_scale * 0.3, bounds)
            new = self.guard.clamp_params(proposed, agent.params)
            if new != agent.params:
                events.append({"event": "tune", "agent": aid,
                               "before": agent.params.to_dict(),
                               "after": new.to_dict()})
                agent.params = new

        # replace the worst with mutated clones of elites
        start_rep = reputation.mean()
        for aid in worst:
            parent = self.agents[self.rng.choice(elites)]
            proposed = mutate(parent.params, self.rng, self.mutation_rate,
                              self.mutation_scale, bounds)
            child_params = self.guard.clamp_params(proposed, parent.params)
            retired = self.agents.pop(aid)
            reputation.remove(aid)
            child = self.spawn(parent.name, child_params,
                               generation=parent.generation + 1,
                               parent=parent.id)
            reputation.register(child.id, init=start_rep)
            events.append({"event": "replace", "retired": retired.id,
                           "new": child.id, "parent": parent.id,
                           "params": child_params.to_dict()})

        self.guard.check_population(len(self.agents))
        return events

    # ----- persistence -----
    def to_dict(self) -> dict:
        return {"counter": self._counter,
                "agents": [a.to_dict() for a in self.agents.values()]}

    def load(self, data: dict) -> None:
        self._counter = data["counter"]
        self.agents = {}
        for a in data["agents"]:
            agent = HeuristicAgent(a["id"], a["name"],
                                   AgentParams.from_dict(a["params"]),
                                   a["generation"], a["parent"])
            self.agents[agent.id] = agent
        self.guard.check_population(len(self.agents))
