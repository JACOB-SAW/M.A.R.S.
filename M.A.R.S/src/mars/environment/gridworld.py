from mars.environment.base import Environment

ACTIONS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


class Gridworld(Environment):
    """Grid with fixed hazards and a goal that respawns when reached."""

    def __init__(self, size: int, n_hazards: int, rng, sensor_radius: int = 3):
        self.size = size
        self.rng = rng
        self.sensor_radius = sensor_radius
        self.actions = ACTIONS
        self.start = (0, 0)
        cells = [(x, y) for x in range(size) for y in range(size)
                 if (x, y) != self.start]
        self.hazards = set(rng.sample(cells, min(n_hazards, len(cells) - 2)))
        self.reset()

    def _new_goal(self):
        free = [(x, y) for x in range(self.size) for y in range(self.size)
                if (x, y) not in self.hazards and (x, y) != self.pos]
        return self.rng.choice(free)

    def reset(self) -> dict:
        self.pos = self.start
        self.visits = {self.pos: 1}
        self.goal = self._new_goal()
        return self.state()

    def state(self) -> dict:
        x, y = self.pos
        near = tuple(h for h in self.hazards
                     if abs(h[0] - x) + abs(h[1] - y) <= self.sensor_radius)
        return {"pos": self.pos, "goal": self.goal, "size": self.size,
                "hazards": near, "visits": self.visits}

    def step(self, action: str):
        dx, dy = self.actions[action]
        nx = min(max(self.pos[0] + dx, 0), self.size - 1)
        ny = min(max(self.pos[1] + dy, 0), self.size - 1)
        reward = -0.1
        info = {"goal": 0, "hazard": 0, "new_cell": 0}

        if (nx, ny) in self.hazards:
            reward -= 5.0
            info["hazard"] = 1          # bounced back, stays in place
        else:
            self.pos = (nx, ny)
            if self.pos not in self.visits:
                info["new_cell"] = 1
            self.visits[self.pos] = self.visits.get(self.pos, 0) + 1
            if self.pos == self.goal:
                reward += 10.0
                info["goal"] = 1
                self.goal = self._new_goal()
        return self.state(), reward, False, info
