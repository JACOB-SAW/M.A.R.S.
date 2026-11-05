import json
from pathlib import Path


class SnapshotStore:
    """Saves the system's state each cycle so you can restore an earlier one."""

    def __init__(self, directory, keep: int = 20):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep = keep

    def save(self, cycle: int, state: dict) -> Path:
        path = self.dir / f"snapshot_{cycle:06d}.json"
        path.write_text(json.dumps(state, indent=2, default=str))
        for old in sorted(self.dir.glob("snapshot_*.json"))[:-self.keep]:
            old.unlink()
        return path

    def load(self, cycle: int) -> dict:
        return json.loads((self.dir / f"snapshot_{cycle:06d}.json").read_text())

    def load_latest(self):
        files = sorted(self.dir.glob("snapshot_*.json"))
        return json.loads(files[-1].read_text()) if files else None
