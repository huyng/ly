"""Tests for session persistence: file layout of new sessions and the
listing filter that keeps foreign session formats out of `/sessions`."""

import json
import os
import re
from datetime import datetime
from pathlib import Path

import pytest

from ly.messages import (
    AssistantMessage,
    ToolResultMessage,
    UserMessage,
    deserialize_message,
    serialize_message,
)
from ly.session import SessionManager


def touch(path: Path, lines: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines or []) + "\n", encoding="utf-8")
    return path


def test_new_session_uses_dated_dir_and_12_hex_name(tmp_path):
    manager = SessionManager(directory=tmp_path)
    manager.new_session()

    assert manager.file is not None
    assert manager.file.parent.name == datetime.now().strftime("%Y-%m-%d")
    assert re.fullmatch(r"[0-9a-f]{12}", manager.file.stem)
    assert manager.file.parent.is_dir()
    # created lazily on first append, so nothing on disk yet
    assert not manager.file.exists()


def test_list_sessions_ignores_foreign_layouts(tmp_path):
    ours = touch(tmp_path / "2026-10-20" / "02c4deaded60.jsonl")
    # a project-dir session from the full ly CLI (32-hex id + index file)
    touch(tmp_path / "home-huyng-ly-32d852" / "9949e2acc878490ca6def010b1fc1c5b.jsonl")
    touch(tmp_path / "home-huyng-ly-32d852" / "index.jsonl")
    # wrong-shaped names inside a date dir are still not ours
    touch(tmp_path / "2026-10-21" / "9949e2acc878490ca6def010b1fc1c5b.jsonl")
    touch(tmp_path / "2026-10-21" / "notasession.jsonl")

    assert SessionManager.list_sessions(tmp_path) == [ours]


def test_list_sessions_newest_first(tmp_path):
    older = touch(tmp_path / "2026-10-01" / "aaaaaaaaaaaa.jsonl")
    newer = touch(tmp_path / "2026-10-02" / "bbbbbbbbbbbb.jsonl")
    os.utime(older, (1_000_000, 1_000_000))
    os.utime(newer, (2_000_000, 2_000_000))

    assert SessionManager.list_sessions(tmp_path) == [newer, older]


def test_list_sessions_missing_directory(tmp_path):
    assert SessionManager.list_sessions(tmp_path / "nope") == []


def test_round_trip_reload(tmp_path):
    manager = SessionManager(directory=tmp_path)
    manager.new_session()
    manager.append(UserMessage("hello"))

    assert manager.file is not None
    assert SessionManager.list_sessions(tmp_path) == [manager.file]

    restored = SessionManager(directory=tmp_path)
    restored.load(manager.file)
    branch = restored.active_branch()
    assert len(branch) == 1
    assert isinstance(branch[0], UserMessage)
    assert branch[0].content == "hello"


@pytest.mark.parametrize(
    "message",
    [
        UserMessage("hello"),
        AssistantMessage(content=[], model="fake"),
        ToolResultMessage(tool_call_id="call", tool_name="read", output="hello"),
    ],
)
def test_message_timestamp_round_trip(message, monkeypatch):
    monkeypatch.setattr("ly.messages.time.time", lambda: 0.999)
    message.timestamp = 123
    saved = serialize_message(message)

    assert deserialize_message(saved).timestamp == 123
    saved.pop("timestamp")
    assert deserialize_message(saved).timestamp == 999


def test_assistant_metadata_round_trip(tmp_path):
    manager = SessionManager(directory=tmp_path)
    manager.new_session()
    message = AssistantMessage(
        content=[],
        model="model",
        api="openai-responses",
    )
    manager.append(message)
    assert manager.file is not None
    assert "provider" not in serialize_message(message)

    restored = SessionManager(directory=tmp_path)
    restored.load(manager.file)
    assert restored.active_branch() == [message]


@pytest.mark.parametrize(
    ("provider", "api"),
    [
        ("openai", "openai-completions"),
        ("openai-responses", "openai-responses"),
        ("fake", ""),
    ],
)
def test_legacy_assistant_metadata_loads(tmp_path, provider, api):
    message = AssistantMessage(content=[], model="model")
    saved = serialize_message(message)
    saved["provider"] = provider
    saved.pop("api")
    path = touch(
        tmp_path / "legacy.jsonl",
        [
            json.dumps(
                {
                    "id": "message",
                    "parent": None,
                    "message": saved,
                }
            ),
            json.dumps({"current": "message"}),
        ],
    )
    manager = SessionManager(directory=tmp_path)
    manager.load(path)
    restored = manager.active_branch()[0]

    assert isinstance(restored, AssistantMessage)
    assert restored.api == api
    assert "provider" not in serialize_message(restored)


def test_active_branch_preserves_timestamps(monkeypatch):
    manager = SessionManager(in_memory=True)
    message = UserMessage("hello")
    message.timestamp = 123
    manager.append(message)
    monkeypatch.setattr("ly.messages.time.time", lambda: 0.999)

    assert manager.active_branch()[0].timestamp == 123
    assert manager.active_branch()[0].timestamp == 123


def test_load_skips_malformed_lines(tmp_path):
    manager = SessionManager(directory=tmp_path)
    manager.new_session()
    manager.append(UserMessage("hello"))
    assert manager.file is not None
    with open(manager.file, "a", encoding="utf-8") as f:
        f.write('{"id": "truncated"\n')
        f.write("not json at all\n")

    restored = SessionManager(directory=tmp_path)
    restored.load(manager.file)
    branch = restored.active_branch()
    assert len(branch) == 1
    assert isinstance(branch[0], UserMessage)
    assert branch[0].content == "hello"


def test_synopsis_is_the_last_prompt_on_one_line(tmp_path):
    manager = SessionManager(directory=tmp_path)
    manager.new_session()
    manager.append(UserMessage("an old prompt"))
    manager.append(UserMessage("  fix the\n  failing tests "))
    assert manager.file is not None

    assert SessionManager.synopsis(manager.file) == "fix the failing tests"


def test_synopsis_is_cut_short_with_an_ellipsis(tmp_path):
    manager = SessionManager(directory=tmp_path)
    manager.new_session()
    manager.append(UserMessage("word " * 100))
    assert manager.file is not None

    synopsis = SessionManager.synopsis(manager.file)

    assert len(synopsis) <= 60
    assert synopsis.endswith("…")


def test_synopsis_skips_damaged_lines_and_files_without_prompts(tmp_path):
    manager = SessionManager(directory=tmp_path)
    manager.new_session()
    manager.append(UserMessage("kept"))
    assert manager.file is not None
    with open(manager.file, "a", encoding="utf-8") as f:
        f.write('{"id": "truncated"\n')
    empty = touch(tmp_path / "2026-10-20" / "00aa00aa00aa.jsonl")

    assert SessionManager.synopsis(manager.file) == "kept"
    assert SessionManager.synopsis(empty) == "(no prompt)"
    assert SessionManager.synopsis(tmp_path / "missing.jsonl") == "(no prompt)"


def test_load_stops_at_dangling_current_pointer(tmp_path):
    manager = SessionManager(directory=tmp_path)
    manager.new_session()
    manager.append(UserMessage("hello"))
    assert manager.file is not None
    with open(manager.file, "a", encoding="utf-8") as f:
        f.write(json.dumps({"current": "missing"}) + "\n")

    restored = SessionManager(directory=tmp_path)
    restored.load(manager.file)
    assert restored.active_branch() == []
