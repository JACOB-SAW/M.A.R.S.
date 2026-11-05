import json
import time
from pathlib import Path


class Logger:
    """Writes every event to a JSONL file and optionally echoes to console."""

    def __init__(self, log_dir, run_id: str | None = None, echo: bool = True):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
        self.path = Path(log_dir) / f"run_{run_id}.jsonl"
        self.echo = echo

    def event(self, kind: str, **data) -> None:
        record = {"t": round(time.time(), 3), "kind": kind, **data}
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def info(self, message: str) -> None:
        if self.echo:
            print(message)
