import random
from pathlib import Path

import yaml

from mars.agents.params import AgentParams
from mars.main import Simulation
from mars.safety.guardrails import Guardrails

ROOT = Path(__file__).resolve().parents[1]


def guard():
    return Guardrails(yaml.safe_load((ROOT / "config" / "protected.yaml").read_text()))


def test_params_are_clamped_to_bounds():
    out = guard().clamp_params(AgentParams(risk_tolerance=5.0, exploration=-3.0))
    assert out.risk_tolerance == 1.0
    assert out.exploration == 0.0


def test_param_change_per_cycle_is_capped():
    old = AgentParams(risk_tolerance=0.5)
    new = guard().clamp_params(AgentParams(risk_tolerance=1.0), old)
    assert abs(new.risk_tolerance - 0.5) <= 0.15 + 1e-9


def test_simulation_runs_and_keeps_population_size(tmp_path):
    (tmp_path / "config").mkdir()
    for f in ("default.yaml", "protected.yaml"):
        (tmp_path / "config" / f).write_text((ROOT / "config" / f).read_text())
    sim = Simulation(tmp_path, seed=1, echo=False)
    n = len(sim.population.agents)
    sim.run(20)
    assert len(sim.population.agents) == n
    assert sim.purposes.adopted is not None


def test_deny_mode_never_changes_purpose_to_unapproved(tmp_path):
    (tmp_path / "config").mkdir()
    for f in ("default.yaml", "protected.yaml"):
        (tmp_path / "config" / f).write_text((ROOT / "config" / f).read_text())
    sim = Simulation(tmp_path, seed=1, approval="deny", echo=False)
    sim.run(20)
    assert sim.purposes.adopted.name == "reach_goals"  # safe fallback
