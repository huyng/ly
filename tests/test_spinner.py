"""Tests for the terminal spinner."""

import time

from ly.spinner import ERASE_LINE, FRAMES, HIDE_CURSOR, SHOW_CURSOR, Spinner


class FakeTty:
    def __init__(self):
        self.writes: list[str] = []
        self._isatty = True

    def isatty(self) -> bool:
        return self._isatty

    def write(self, text: str) -> None:
        self.writes.append(text)

    def flush(self) -> None:
        pass


def test_spinner_noop_without_tty():
    stream = FakeTty()
    stream._isatty = False
    spinner = Spinner(stream=stream)
    spinner.start("glm-4.6")
    time.sleep(0.03)
    spinner.stop()
    assert stream.writes == []
    assert not spinner.active


def test_spinner_tty_lifecycle():
    stream = FakeTty()
    spinner = Spinner(stream=stream, interval=0.01)
    spinner.start("glm-4.6")
    time.sleep(0.06)  # a few frames
    assert spinner.active
    spinner.stop()

    out = "".join(stream.writes)
    assert HIDE_CURSOR in out
    assert "glm-4.6" in out
    assert any(frame in out for frame in FRAMES)
    assert out.endswith(ERASE_LINE + SHOW_CURSOR)

    # stop is idempotent and the thread is joined
    before = len(stream.writes)
    spinner.stop()
    assert len(stream.writes) == before
    assert spinner._thread is None


def test_spinner_label_update_while_active():
    stream = FakeTty()
    spinner = Spinner(stream=stream, interval=0.01)
    spinner.start("first")
    spinner.start("second")  # updates label, keeps one thread
    time.sleep(0.04)
    spinner.stop()
    out = "".join(stream.writes)
    assert "second" in out
