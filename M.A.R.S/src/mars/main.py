import argparse
import random
from pathlib import Path

import yaml

from mars.council.arbiter import Arbiter
from mars.council.population import Population
from mars.council.reputation import Reputation
from mars.environment.gridworld import Gridworld
from mars.purpose.evaluator import PurposeEvaluator
from mars.purpose.hypotheses import default_hypotheses
from mars.purpose.manager import PurposeManager
from mars.safety.approval import Approver
from mars.safety.guardrails import Guardrails
from mars.safety.rollback import SnapshotStore
from mars.telemetry.logger import Logger


def load_yaml(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class Simulation:
    def __init__(self, root: Path, seed=None, approval="auto", echo=True):
        cfg = load_yaml(root / "config" / "default.yaml")
        protected = load_yaml(root / "config" / "protected.yaml")
        self.cfg = cfg
        self.rng = random.Random(cfg["seed"] if seed is None else seed)

        self.logger = Logger(root / "data" / "logs", echo=echo)
        self.guard = Guardrails(protected)
        self.snapshots = SnapshotStore(root / "data" / "snapshots")
        self.approver = Approver(approval, self.logger)

        env = cfg["environment"]
        self.steps = self.guard.cap_steps(env["steps_per_cycle"])
        self.env = Gridworld(env["size"], env["hazards"], self.rng)

        rep = cfg["reputation"]
        self.reputation = Reputation(rep["decay"], rep["min_weight"], rep["sensitivity"])

        pop = cfg["population"]
        self.population = Population(self.guard, self.rng, pop["elite_fraction"],
                                     pop["mutation_rate"], pop["mutation_scale"])
        self.population.add_from_templates(cfg["agent_templates"])
        for aid in self.population.agents:
            self.reputation.register(aid)

        self.arbiter = Arbiter()
        pur = cfg["purpose"]
        self.purposes = PurposeManager(
            default_hypotheses(), PurposeEvaluator(), self.approver, self.rng,
            pur["trial_cycles"], pur["exploit_cycles"])
        self.cycle = 0

    # ----- one cycle = one episode -----
    def run_cycle(self) -> dict:
        purpose = self.purposes.current
        state = self.env.reset()
        agents = list(self.population.agents.values())
        weights = self.reputation.weights()

        raw_reward = 0.0
        goals = hazards = 0
        for _ in range(self.steps):
            proposals = [a.propose(state, self.env.actions, self.rng) for a in agents]
            action, _tally = self.arbiter.decide(proposals, weights, self.rng)
            state, reward, _done, info = self.env.step(action)

            raw_reward += reward
            goals += info["goal"]
            hazards += info["hazard"]
            self.reputation.credit_step(proposals, action, purpose.reward(info))

        return {
            "raw_reward": round(raw_reward, 3),
            "goals": goals,
            "hazards": hazards,
            "hazard_rate": hazards / self.steps,
            "coverage": round(len(self.env.visits) / self.env.size ** 2, 3),
        }

    def snapshot_state(self) -> dict:
        return {
            "cycle": self.cycle,
            "population": self.population.to_dict(),
            "reputation": self.reputation.to_dict(),
            "purpose": self.purposes.current.to_dict(),
        }

    def resume(self) -> bool:
        snap = self.snapshots.load_latest()
        if not snap:
            return False
        self.population.load(snap["population"])
        self.reputation.load(snap["reputation"])
        self.cycle = snap["cycle"]
        return True

    def run(self, cycles: int) -> None:
        cycles = self.guard.cap_cycles(cycles)
        for _ in range(cycles):
            self.cycle += 1
            purpose_name = self.purposes.current.name
            metrics = self.run_cycle()
            scores = self.reputation.end_cycle()
            evo_events = self.population.evolve(scores, self.reputation)
            purpose_events = self.purposes.on_cycle_end(metrics)

            self.logger.event("cycle", cycle=self.cycle, purpose=purpose_name,
                              **metrics)
            for e in evo_events + purpose_events:
                rest = {k: v for k, v in e.items() if k != "event"}
                self.logger.event(e["event"], cycle=self.cycle, **rest)
            self.snapshots.save(self.cycle, self.snapshot_state())

            top = max(self.reputation.ema, key=self.reputation.ema.get)
            top_agent = self.population.agents[top]
            self.logger.info(
                f"cycle {self.cycle:4d} | purpose={purpose_name:<12} "
                f"reward={metrics['raw_reward']:7.1f} goals={metrics['goals']:2d} "
                f"hazards={metrics['hazards']:2d} cover={metrics['coverage']:.2f} "
                f"| top={top_agent.name} (gen {top_agent.generation})")
            for e in purpose_events:
                if e["event"] in ("purpose_adopted", "purpose_candidate"):
                    self.logger.info(f"           >> {e['event']}: {e['purpose']['name']} "
                                     f"{e['purpose']['weights']}")


def cli():
    ap = argparse.ArgumentParser(description="M.A.R.S. simulation")
    ap.add_argument("--cycles", type=int, default=60)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--root", type=Path, default=Path.cwd(),
                    help="project root containing config/ and data/")
    ap.add_argument("--approval", choices=["auto", "ask", "deny"], default="auto",
                    help="how purpose changes are approved")
    ap.add_argument("--resume", action="store_true",
                    help="restore the latest snapshot first")
    args = ap.parse_args()

    sim = Simulation(args.root, seed=args.seed, approval=args.approval)
    if args.resume and sim.resume():
        sim.logger.info(f"Resumed from cycle {sim.cycle}")
    sim.run(args.cycles)


if __name__ == "__main__":
    cli()
