"""System prompt assembly: base instructions plus discovered context files.

Like pi, tau builds the system prompt from base instructions and context
files (AGENTS.md) discovered in the working directory and its ancestors, plus
a user-level AGENTS.md in ~/.agents. When both scopes carry instructions the
prompt says so instead of merging them: a personal default and a project
override can disagree, and hiding that would leave the model silently
following whichever one happened to be concatenated last.
"""

import os

BASE_SYSTEM_PROMPT = """\
You are tau, an expert coding assistant operating inside tau, a coding agent \
harness. You help users by reading files, executing commands, editing code, \
and writing new files.

# Rules

- Use grep and glob to find code instead of `bash` + rg/find/ls.
- Use read to examine files instead of cat or sed.
- Use edit for precise changes. The text to replace must match exactly and be
  unique in the file. When changing multiple separate locations in one file,
  batch them into a single edit call instead of multiple edit calls.
- Use write only for new files or complete rewrites.
- Use subagent when answering needs many search/read calls whose raw output you do
  not need to keep. Give it a self-contained prompt: it cannot ask questions,
  and it cannot edit files.
- Be concise in your responses.
- Show file paths clearly when working with files.

# Environment

Working directory: {cwd}
"""


def find_context_file(cwd: str | None) -> str | None:
    """Find AGENTS.md in cwd or the nearest ancestor, like pi."""
    directory = os.path.abspath(cwd or os.getcwd())
    while True:
        candidate = os.path.join(directory, "AGENTS.md")
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(directory)
        if parent == directory:
            return None
        directory = parent


def read_instructions(path: str | None) -> str:
    """Read a context file; unreadable or blank files count as absent."""
    if not path:
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def context_files(cwd: str | None) -> list[tuple[str, str, str]]:
    """The AGENTS.md files that apply here as (path, scope, text), user first."""
    user_file = os.path.expanduser("~/.agents/AGENTS.md")
    project_file = find_context_file(cwd)
    if project_file == user_file:  # cwd inside ~/.agents: same file, one scope
        project_file = None

    sections: list[tuple[str, str, str]] = []
    if text := read_instructions(user_file):
        sections.append((user_file, "User", text))
    if project_file and (text := read_instructions(project_file)):
        sections.append((project_file, "Project", text))
    return sections


def build_system_prompt(cwd: str | None = None) -> str:
    prompt = BASE_SYSTEM_PROMPT.format(cwd=os.path.abspath(cwd or os.getcwd()))
    sections = context_files(cwd)

    if len(sections) == 2:
        prompt += (
            "\n# Instructions\n\n"
            f"Two AGENTS.md files apply: {sections[0][0]} is the user's personal "
            f"default, {sections[1][0]} is project-level for this repository. "
            "Both are quoted below. Follow the project file wherever it is more "
            "specific; fall back to the user file where it is silent; if they "
            "outright conflict, say so rather than picking silently.\n"
        )
    elif sections:
        prompt += f"\n# {sections[0][1]} instructions\n"

    for path, _, text in sections:
        prompt += f"\nFrom {path}:\n\n{text}\n"
    return prompt
