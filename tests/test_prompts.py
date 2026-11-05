"""prompts.py: user-level and project-level AGENTS.md assembly."""

import pytest

from tau.prompts import build_system_prompt, context_files


def write_context(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


def test_no_context_files(tmp_path, home):
    prompt = build_system_prompt(str(tmp_path))
    assert "# Instructions" not in prompt
    assert "From " not in prompt


def test_project_only(tmp_path, home):
    project = write_context(tmp_path / "AGENTS.md", "run pytest")
    prompt = build_system_prompt(str(tmp_path))
    assert "# Project instructions" in prompt
    assert f"From {project}:" in prompt
    assert "Two AGENTS.md" not in prompt


def test_user_only(tmp_path, home):
    user = write_context(home / ".agents" / "AGENTS.md", "be terse")
    prompt = build_system_prompt(str(tmp_path))
    assert "# User instructions" in prompt
    assert f"From {user}:" in prompt
    assert "Two AGENTS.md" not in prompt


def test_both_scopes_are_named(tmp_path, home):
    user = write_context(home / ".agents" / "AGENTS.md", "be terse")
    project = write_context(tmp_path / "AGENTS.md", "run pytest")
    prompt = build_system_prompt(str(tmp_path))
    assert "Two AGENTS.md files apply" in prompt
    assert f"{user} is the user's personal default" in prompt
    assert f"{project} is project-level" in prompt
    # both bodies present, user first so the project override reads last
    assert prompt.index("be terse") < prompt.index("run pytest")


def test_ancestor_file_is_ignored(tmp_path, home):
    write_context(tmp_path / "AGENTS.md", "outer")
    inner = tmp_path / "service"
    inner.mkdir()
    prompt = build_system_prompt(str(inner))
    assert "outer" not in prompt
    assert "# Instructions" not in prompt


def test_subdirectory_instructions_are_not_mixed_in(tmp_path, home):
    write_context(tmp_path / "AGENTS.md", "outer")
    inner = tmp_path / "service"
    project = write_context(inner / "AGENTS.md", "inner")
    prompt = build_system_prompt(str(inner))
    assert f"From {project}:" in prompt
    assert "outer" not in prompt


def test_blank_project_file_does_not_shadow_user_file(tmp_path, home):
    write_context(home / ".agents" / "AGENTS.md", "be terse")
    write_context(tmp_path / "AGENTS.md", "   \n")
    prompt = build_system_prompt(str(tmp_path))
    assert "# User instructions" in prompt
    assert "be terse" in prompt
    assert "Two AGENTS.md" not in prompt


def test_context_files_reports_user_then_project(tmp_path, home):
    user = write_context(home / ".agents" / "AGENTS.md", "be terse")
    project = write_context(tmp_path / "AGENTS.md", "run pytest")
    found = [(path, scope) for path, scope, _ in context_files(str(tmp_path))]
    assert found == [(str(user), "User"), (str(project), "Project")]


def test_cwd_inside_agents_dir_quotes_file_once(tmp_path, home):
    user = write_context(home / ".agents" / "AGENTS.md", "be terse")
    prompt = build_system_prompt(str(user.parent))
    assert "Two AGENTS.md" not in prompt
    assert prompt.count(f"From {user}:") == 1
