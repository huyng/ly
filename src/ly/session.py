"""Session persistence: append-only JSONL with a tree-shaped entry history.

Modeled after pi's session format: each entry has an id and refers to its
parent, so messages form a tree whose active branch supplies history for the
next model request. The active leaf is recorded with pointer lines; only the
last pointer in the file matters, which keeps the format append-only.
"""

import json
import re
import textwrap
import uuid
from datetime import datetime
from pathlib import Path

from .messages import AgentMessage, deserialize_message, new_id, serialize_message

# The sessions directory is shared with other tools (notably the full ly CLI),
# which lay their sessions out as `<cwd-slug>-<hash>/<uuid4>.jsonl` alongside an
# `index.jsonl`. Only files this module itself writes are ours: a 12-hex id in a
# YYYY-MM-DD directory. Anything else uses a different entry schema and would
# load as an empty session, so it must not show up in the listing.
SESSION_ID_RE = re.compile(r"^[0-9a-f]{12}$")
DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

#: How much of a session's last prompt `/sessions` shows.
SYNOPSIS_CHARS = 60
#: What a session file with no readable prompt is called in the listing.
NO_PROMPT = "(no prompt)"


class SessionManager:
    """One session: the entry tree in memory plus the JSONL file backing it.

    `directory` is where the files go. `None` drops the file side entirely, so
    the tree lives only in `self.entries`; sub-agents and tests use that. The
    path is always supplied by the caller — this module reads no environment
    and holds no default, so where sessions live is one value, resolved once by
    `Settings` and passed down.
    """

    def __init__(self, directory: Path | None = None):
        self.directory = directory
        self.file: Path | None = None  # set by new_session/load, None until then
        self.entries: dict[str, dict] = {}  # id -> entry, all branches, not just active
        self.current_id: str | None = None  # active leaf; history walks its parents

    # -- construction ----------------------------------------------------------

    def new_session(self) -> None:
        """Forget history and choose the file the next append will create.

        The file itself appears on the first `append`, not here.
        """
        self.entries = {}
        self.current_id = None
        if self.directory is None:
            return
        date = datetime.now().strftime("%Y-%m-%d")
        self.directory.joinpath(date).mkdir(parents=True, exist_ok=True)
        self.file = self.directory / date / f"{uuid.uuid4().hex[:12]}.jsonl"

    def load(self, path: Path) -> None:
        """Rebuild the tree from a session file, ending at the last leaf pointer.

        Lines that do not parse are skipped instead of fatal: a session killed
        mid-write leaves a truncated final line.
        """
        self.file = path
        self.entries = {}
        self.current_id = None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a truncated final line must not lose the session
                if not isinstance(data, dict):
                    continue
                # Two line shapes: `{"current": id}` pointers and message entries.
                if "current" in data:
                    self.current_id = data["current"]
                elif "id" in data and "message" in data:
                    self.entries[data["id"]] = data

    # -- queries ---------------------------------------------------------------

    def active_branch(self) -> list[AgentMessage]:
        """Messages from the root down to the current entry."""
        branch: list[AgentMessage] = []
        entry_id = self.current_id
        while entry_id is not None:
            entry = self.entries.get(entry_id)
            if entry is None:
                break
            branch.append(deserialize_message(entry["message"]))
            entry_id = entry.get("parent")
        branch.reverse()
        return branch

    @staticmethod
    def list_sessions(directory: Path | None) -> list[Path]:
        """Session files this module wrote, most recently modified first.

        Args:
            directory: Where sessions are written. `None` is an in-memory
                session, which has no files to list.

        Returns:
            Matching session files, newest first; empty when there is nothing
            to list or the directory does not exist yet.
        """
        if directory is None or not directory.is_dir():
            return []
        sessions = [
            p
            for p in directory.glob("*/*.jsonl")
            if p.is_file()
            and DATE_DIR_RE.match(p.parent.name)
            and SESSION_ID_RE.match(p.stem)
        ]
        return sorted(sessions, key=lambda p: p.stat().st_mtime, reverse=True)

    @staticmethod
    def synopsis(path: Path, chars: int = SYNOPSIS_CHARS) -> str:
        """A one-line summary of the last prompt in a session file.

        The listing is more useful when it says what a session is about right
        now than how it started. Reading backwards makes that the cheap end to
        start from; damaged lines are skipped for the same reason `load` skips
        them: a session killed mid-write must still be listed.

        Args:
            path: A session JSONL file.
            chars: Longest result to return, default `SYNOPSIS_CHARS`. Callers
                pass the room a narrow terminal has left, though the text is cut
                on a word boundary, so the result is usually shorter.

        Returns:
            The last user prompt written to the file, whitespace collapsed to
            one line and cut to `chars`, or `NO_PROMPT` when the file holds no
            prompt or cannot be read.
        """
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return NO_PROMPT

        for line in reversed(lines):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = data.get("message") if isinstance(data, dict) else None
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return textwrap.shorten(content, max(chars, 1), placeholder="…")
        return NO_PROMPT

    # -- writes ----------------------------------------------------------------

    def compact(self, summary: AgentMessage, keep: list[AgentMessage]) -> None:
        """Re-root the active branch at `summary`, then replay `keep`.

        Compaction is a *branch* operation, not a delete: the previous entries
        stay in `self.entries` and in the JSONL file, so the full transcript
        survives on disk. Only the active leaf moves.
        """
        entries = []
        parent = None
        for message in [summary, *keep]:
            entry = self._entry(message, parent)
            entries.append(entry)
            parent = entry["id"]
        self._commit(entries)

    def append(self, message: AgentMessage) -> str:
        """Add `message` as a child of the current leaf and move the leaf to it.

        Returns the new entry id. The pointer line is written after the entry so
        the entry is always on disk before it becomes the active branch.
        """
        entry = self._entry(message, self.current_id)
        self._commit([entry])
        return entry["id"]

    @staticmethod
    def _entry(message: AgentMessage, parent: str | None) -> dict:
        """Build a session entry without changing the active branch."""
        return {
            "id": new_id(),
            "parent": parent,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "message": serialize_message(message),
        }

    def _commit(self, entries: list[dict]) -> None:
        """Write entries and their final pointer before changing memory."""
        # `file` is None for an in-memory session, and stays None until
        # `new_session` picks one for a persisted one.
        if self.file is not None:
            with open(self.file, "a", encoding="utf-8") as f:
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")
                f.write(json.dumps({"current": entries[-1]["id"]}) + "\n")
        for entry in entries:
            self.entries[entry["id"]] = entry
        self.current_id = entries[-1]["id"]
