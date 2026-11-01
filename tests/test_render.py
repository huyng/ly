"""Tests for the terminal presentation layer."""

from tau.render import (
    GutterWriter,
    MarkdownRenderer,
    render_tool_result,
    summarize_tool_call,
)
from tau.theme import Theme, color_enabled


class FakeTty:
    def isatty(self) -> bool:
        return True


def test_markdown_rendering_plain_mode():
    renderer = MarkdownRenderer(Theme(False), width=60)
    out = renderer.feed(
        "## Plan\n"
        "\n"
        "- first step\n"
        "- second step\n"
        "\n"
        "```python\n"
        "x = 1\n"
        "```\n"
    )
    out += renderer.flush()

    assert "Plan" in out
    assert "## " not in out
    assert "• first step" in out
    assert "╭─ python" in out
    assert "│ x = 1" in out
    assert out.endswith("\n")
    assert "\x1b[" not in out  # colors disabled


def test_markdown_rendering_survives_split_deltas():
    renderer = MarkdownRenderer(Theme(False), width=60)
    out = renderer.feed("**bo")
    out += renderer.feed("ld**\n")
    out += renderer.flush()

    assert "bold" in out
    assert "**" not in out


def test_markdown_fence_closes_on_flush():
    renderer = MarkdownRenderer(Theme(False), width=60)
    out = renderer.feed("```\nunfinished")
    out += renderer.flush()

    assert "│ unfinished" in out
    assert "╰─" in out
    assert out.endswith("\n")


def test_colors_are_emitted_when_enabled():
    out = MarkdownRenderer(Theme(True), width=60).feed("`code` and **bold**\n")
    assert "\x1b[" in out


def test_gutter_stays_continuous_across_blank_lines():
    writer = GutterWriter(Theme(False))
    out = writer.feed("one\n\ntwo") + writer.flush()

    assert out == "│ one\n│\n│ two\n"


def test_gutter_holds_across_blank_line_split_over_deltas():
    writer = GutterWriter(Theme(False))
    out = writer.feed("one\n") + writer.feed("\n") + writer.feed("two") + writer.flush()

    assert out == "│ one\n│\n│ two\n"


def test_gutter_blank_line_is_dimmed_without_trailing_space():
    out = GutterWriter(Theme(True)).feed("one\n\ntwo")

    assert "\x1b[2m│\x1b[0m\n" in out


def test_gutter_wraps_long_lines_at_width():
    writer = GutterWriter(Theme(False), width=20)
    out = writer.feed("x" * 45) + writer.flush()

    # One column under the width, so the terminal never wraps the line itself.
    assert out == "│ " + "x" * 17 + "\n│ " + "x" * 17 + "\n│ " + "x" * 11 + "\n"


def test_gutter_breaks_on_spaces_not_mid_word():
    writer = GutterWriter(Theme(False), width=20)
    out = writer.feed("alpha beta gamma delta\n")

    assert out == "│ alpha beta gamma\n│ delta\n"


def test_gutter_wraps_when_a_delta_lands_on_the_margin():
    writer = GutterWriter(Theme(False), width=20)
    out = writer.feed("x" * 18) + writer.feed("y") + writer.flush()

    assert out == "│ " + "x" * 17 + "\n│ xy\n"


def test_gutter_frames_a_labelled_stream():
    writer = GutterWriter(Theme(False), label="thinking")
    out = writer.feed("one") + writer.feed("\ntwo") + writer.flush()

    assert out == "╭─ thinking\n│ one\n│ two\n╰─\n"


def test_gutter_frame_closes_after_a_trailing_newline():
    writer = GutterWriter(Theme(False), label="thinking")
    out = writer.feed("one\n") + writer.flush()

    assert out == "╭─ thinking\n│ one\n╰─\n"


def test_gutter_drops_blank_lines_before_the_header():
    writer = GutterWriter(Theme(False), label="thinking")
    out = writer.feed("\n\none") + writer.flush()

    assert out == "╭─ thinking\n│ one\n╰─\n"


def test_gutter_flush_is_silent_when_nothing_streamed():
    assert GutterWriter(Theme(False), label="thinking").flush() == ""


def test_diff_tool_result_is_colorized_and_summarized():
    output = (
        "Edited src/app.py (1 change)\n\n"
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-old line\n"
        "+new line\n"
    )
    out = render_tool_result(Theme(False), "edit", output, False, 0.42)

    assert "✓" in out
    assert "+1" in out and "−1" in out
    assert "-old line" in out
    assert "+new line" in out


def test_bash_call_prints_the_whole_script():
    script = 'for f in *; do\n  echo "$f"\ndone'
    out = summarize_tool_call(Theme(False), "bash", {"command": script}, 40)

    assert out.splitlines() == ["│ for f in *; do", '│   echo "$f"', "│ done"]
    # A long one-liner is a script too: no ellipsis, no width cut.
    long = "echo " + "x" * 200
    assert summarize_tool_call(Theme(False), "bash", {"command": long}, 40) == long


def test_bash_result_shows_preview_and_exit_code():
    out = render_tool_result(
        Theme(False), "bash", "hello\nworld\n[exit code: 0]", False, 1.2
    )
    assert "hello" in out
    assert out == ("├─ ✓ done 1.2s ↓\n│ hello\n│ world\n│ [exit code: 0]\n╰─")


def test_tool_result_status_and_output_frame():
    for name in ("bash", "read", "write", "grep", "edit"):
        for is_error, status in ((False, "✓ done"), (True, "✗ error")):
            out = render_tool_result(Theme(False), name, "hello", is_error, 1.4)
            assert out.startswith(f"├─ {status} 1.4s ↓\n│ ")
            assert out.endswith("\n╰─")
            assert "\x1b[" not in out


def test_tool_result_status_colors():
    for is_error, status, color in (
        (False, "✓ done", "\x1b[32m"),
        (True, "✗ error", "\x1b[31m"),
    ):
        out = render_tool_result(Theme(True), "bash", "hello", is_error, 1.4)
        glyph, label = status.split()
        assert f"{color}{glyph}\x1b[0m" in out
        assert f"\x1b[2m {label} 1.4s ↓\x1b[0m" in out


def test_empty_tool_result_closes_frame():
    for is_error in (False, True):
        out = render_tool_result(Theme(False), "bash", "", is_error, 1.4)
        assert out.splitlines()[1:] == ["╰─"]


def test_read_result_is_summarized_not_dumped():
    out = render_tool_result(Theme(False), "read", "line\n" * 500, False, 0.01)
    assert "500 lines" in out
    assert "line\n" not in out


def test_error_result_is_red_and_bounded():
    out = render_tool_result(
        Theme(False),
        "bash",
        "Error: boom\n" + "\n".join(f"line {i}" for i in range(20)),
        True,
        0.1,
    )
    assert "✗" in out
    assert "Error: boom" in out
    assert "more lines" in out


def test_color_enabled_respects_no_color(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    assert not color_enabled(FakeTty())


def test_color_enabled_requires_tty(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")

    class Pipe:
        def isatty(self) -> bool:
            return False

    assert not color_enabled(Pipe())
    assert color_enabled(FakeTty())
