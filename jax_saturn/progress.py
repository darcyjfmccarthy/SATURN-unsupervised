"""Terminal heartbeat and separate, append-only execution timing records."""

from contextlib import contextmanager
from contextvars import ContextVar
import json
from pathlib import Path
import threading
import time
from functools import wraps

CURRENT_PROGRESS = ContextVar("saturn_progress", default=None)


def training_progress(function):
    """Give standalone foundation stages the same timing records as fine-tuning."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        if CURRENT_PROGRESS.get() is not None:
            return function(*args, **kwargs)
        with Progress(Path(kwargs['output_dir']) / 'execution.jsonl'):
            return function(*args, **kwargs)
    return wrapped


class Progress:
    def __init__(self, path=None, interval=25):
        self.path = None if path is None else Path(path)
        self.interval = interval
        self.active = "starting"
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.lock = threading.Lock()

    def event(self, event, **values):
        record = {"time_unix": time.time(), "event": event, **values}
        line = json.dumps(record, sort_keys=True, allow_nan=False)
        with self.lock:
            print(line, flush=True)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a") as handle:
                    handle.write(line + "\n")
        return record

    def __enter__(self):
        self.token = CURRENT_PROGRESS.set(self)
        def heartbeat():
            while not self.stop.wait(self.interval):
                self.event("heartbeat", stage=self.active, elapsed_seconds=time.monotonic() - self.started)
        self.thread = threading.Thread(target=heartbeat, name="saturn-heartbeat", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=1)
        CURRENT_PROGRESS.reset(self.token)

    @contextmanager
    def phase(self, stage, **values):
        previous = self.active
        self.active = stage
        self.event("stage_start", stage=stage, **values)
        started = time.monotonic()
        try:
            yield
        finally:
            self.event("stage_end", stage=stage, seconds=time.monotonic() - started, **values)
            self.active = previous
