"""Terminal spinner shown while waiting for the model's first response chunk.

The agent loop runs synchronously in the main thread, so the spinner
animates from a daemon thread. All writes are serialized through a lock
shared with `stop()`, which erases the spinner line and restores the cursor
before real output is printed. No-ops when stdout is not a TTY.
"""

import itertools
import sys
import threading
import time

FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
HIDE_CURSOR = "\x1b[?25l"
SHOW_CURSOR = "\x1b[?25h"
ERASE_LINE = "\r\x1b[K"
DIM = "\x1b[2m"
RESET = "\x1b[0m"


class Spinner:
    def __init__(self, stream=None, interval: float = 0.08):
        self.stream = stream if stream is not None else sys.stdout
        self.interval = interval
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._active = False
        self._start = 0.0
        self._frames = itertools.cycle(FRAMES)
        self._label = ""

    @property
    def active(self) -> bool:
        return self._active

    def start(self, label: str = "") -> None:
        """Start spinning with a label; updates the label if already active."""
        if not getattr(self.stream, "isatty", lambda: False)():
            return
        with self._lock:
            if self._active:
                self._label = label
                return
            self._active = True
            self._start = time.monotonic()
            self._label = label
        self.stream.write(HIDE_CURSOR)
        self.stream.flush()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop spinning and erase the spinner line. Idempotent."""
        with self._lock:
            if not self._active:
                return
            self._active = False
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self.stream.write(ERASE_LINE + SHOW_CURSOR)
        self.stream.flush()

    # -- internals ---------------------------------------------------------------

    def _spin(self) -> None:
        while True:
            with self._lock:
                if not self._active:
                    return
                elapsed = time.monotonic() - self._start
                frame = next(self._frames)
                self.stream.write(
                    f"\r{DIM}{frame} {elapsed:>4.0f}s {self._label}{RESET}\x1b[K"
                )
                self.stream.flush()
            time.sleep(self.interval)
