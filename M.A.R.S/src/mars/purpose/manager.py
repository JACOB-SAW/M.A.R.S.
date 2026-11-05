from mars.purpose.hypotheses import make_variant


class PurposeManager:
    """Runs the search for a purpose.

    1. TRIAL: act under each candidate purpose for a few cycles.
    2. Score each with the fixed anchor criteria.
    3. If the winner differs from the adopted purpose, ask for approval.
    4. EXPLOIT the adopted purpose for a while.
    5. Generate a new variant, trial it against the adopted one, repeat.
    """

    def __init__(self, pool, evaluator, approver, rng,
                 trial_cycles=4, exploit_cycles=8, max_pool=8):
        self.pool = list(pool)
        self.evaluator = evaluator
        self.approver = approver
        self.rng = rng
        self.trial_cycles = trial_cycles
        self.exploit_cycles = exploit_cycles
        self.max_pool = max_pool
        self._variant_count = 0

        self.adopted = None
        self.phase = "trial"
        self.contenders = [p.name for p in self.pool]
        self.queue = list(self.pool)
        self.current = self.queue.pop(0)
        self.left = trial_cycles

    def _by_name(self, name):
        return next(p for p in self.pool if p.name == name)

    def on_cycle_end(self, metrics: dict) -> list:
        """Call once per cycle. Returns a list of events to log."""
        self.evaluator.record(self.current.name, metrics)
        self.left -= 1
        if self.left > 0:
            return []
        return self._advance()

    def _advance(self) -> list:
        events = []
        if self.phase == "trial":
            if self.queue:
                self.current = self.queue.pop(0)
                self.left = self.trial_cycles
                return events

            best_name, scores = self.evaluator.best(self.contenders, self.trial_cycles)
            events.append({"event": "purpose_scores", "scores": scores,
                           "winner": best_name})
            winner = self._by_name(best_name)
            if self.adopted is None or winner.name != self.adopted.name:
                before = self.adopted.name if self.adopted else "none"
                ok = self.approver.request(
                    f"Change purpose: {before} -> {winner.name} "
                    f"({winner.description}) weights={winner.weights}")
                if ok:
                    self.adopted = winner
                    events.append({"event": "purpose_adopted",
                                   "purpose": winner.to_dict()})
                else:
                    events.append({"event": "purpose_denied",
                                   "purpose": winner.name})
                    if self.adopted is None:
                        self.adopted = self.pool[0]
            self.current = self.adopted
            self.phase = "exploit"
            self.left = self.exploit_cycles
        else:
            self._variant_count += 1
            variant = make_variant(self.adopted, self.rng, self._variant_count)
            if len(self.pool) >= self.max_pool:
                spare = [p for p in self.pool if p.name != self.adopted.name]
                self.pool.remove(spare[0])
            self.pool.append(variant)
            events.append({"event": "purpose_candidate",
                           "purpose": variant.to_dict()})
            self.contenders = [self.adopted.name, variant.name]
            self.queue = []
            self.current = variant
            self.phase = "trial"
            self.left = self.trial_cycles
        return events
