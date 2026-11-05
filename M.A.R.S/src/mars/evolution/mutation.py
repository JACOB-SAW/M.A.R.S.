from mars.agents.params import AgentParams


def mutate(params: AgentParams, rng, rate: float, scale: float,
           param_bounds: dict) -> AgentParams:
    """Randomly nudge each param with probability `rate`.
    `scale` is a fraction of each param's allowed range."""
    out = params.to_dict()
    for name, value in out.items():
        if rng.random() < rate:
            lo, hi = param_bounds[name]
            out[name] = value + rng.gauss(0, scale * (hi - lo))
    return AgentParams.from_dict(out)
