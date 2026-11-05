from mars.agents.params import AgentParams


def apply_bounds(new: AgentParams, old: AgentParams | None,
                 param_bounds: dict, max_delta: float) -> AgentParams:
    """Clamp every param into its allowed range, and (if `old` is given)
    limit how far it can move from `old` in a single step."""
    out = {}
    for name, value in new.to_dict().items():
        lo, hi = param_bounds[name]
        if old is not None:
            step = max_delta * (hi - lo)
            prev = getattr(old, name)
            value = min(max(value, prev - step), prev + step)
        out[name] = min(max(value, lo), hi)
    return AgentParams.from_dict(out)
