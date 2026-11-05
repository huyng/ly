# τ tau

A minimal, extensible AI coding agent for the terminal, written in Python and modeled after [pi](https://pi.dev). It transparently runs sandboxed inside Docker by default.

When you run `tau` it automatically forks and launches itself inside a docker container mounted with your current working directory.

tau replicates pi's core architecture — the agent loop, streaming events,
session persistence, tool execution, and context-file discovery — in roughly
3000 lines of dependency-free Python code.

## Installation & Usage

Requires Python 3.12+. There are no runtime dependencies — installation pulls nothing beyond the standard library.

```bash
uv tool install git+https://github.com/huyng/tau
```

Point tau to an LLM provider by setting environment variables. It is compatible with any OpenAI-API compatible endpoint.

```bash
export TAU_API_KEY=sk-              # API key for the selected provider
export TAU_MODEL=gpt-4.1            # The model you would like to use
export TAU_BASE_URL=
```

## Use

```bash
tau                                # interactive REPL in the current directory
tau "fix the failing tests"        # start with an initial prompt
tau -p "what does this repo do?"   # print mode: run once, write final response
tau --resume 2                     # resume a previous session
```

### Sandboxing with Docker

By default, `tau` runs inside a Docker container by forking and re-executing itself as `docker run`, mounting the working directory at `/workspace`, so the tools can only touch the project. Sessions are bind-mounted from the host's `~/.tau/sessions`, and `~/.agents` is mounted read-only.

The container runs as your host uid/gid, so files it writes into the project
belong to you. Its `HOME` moves to `/tmp`, because a non-root uid cannot write
the image's `/root`; anything the `bash` tool sets up via `~` (a `~/.gitconfig`, say) therefore lasts only for that container.

```bash
export TAU_DOCKER_IMAGE=python:3.12       # image to use (default python:3.12)
tau
```

The default image is `python:3.12`, which carries git, make, gcc and curl, so
the `bash` tool has the usual build tools. Point `TAU_DOCKER_IMAGE` at another
image when you need more; it must provide `python3` at version 3.12 or newer,
since that is what tau runs on inside it. The startup banner names the image in
use (`• docker python:3.12 · /workspace`), so the container is never a mystery.
In print mode the same fact goes to stderr (`tau: docker python:3.12`) and
stdout carries only the answer, so `tau -p … | jq` keeps working.

You can run tau without a sandbox with the `--nosandbox` flag.

```bash
tau --nosandbox     # stay on the host (or TAU_DOCKER=0)
```

## How it works

A session is tau's record of a conversation. The pieces mirror pi:

- **Agent loop** — a submitted message is added to the active branch; tau
  builds a model request from the system prompt, active branch, and tool
  definitions; the provider streams an assistant response containing text
  and/or tool calls; tau executes each tool call and records the results.
  Tool results trigger another turn; otherwise the run ends.
- **Backend** — one request per turn — `chat/completions` by default, or
  `/v1/responses` for models that refuse tools on chat-completions — streamed over
  `urllib` and parsed line by line as SSE, so there is no client SDK to depend
  on. The backend holds a `Provider` (model, endpoint, credentials) and reads it
  on every request, so a model switch takes effect immediately. Text, reasoning,
  and tool-call deltas accumulate into the finished message.
- **Events** — subscribe with `session.subscribe(callback)` to observe
  message start/end, text deltas, tool execution, and run lifecycle.
- **Sessions** — append-only JSONL under `~/.tau/sessions/<date>/` (override
  with `TAU_SESSIONS_DIR`). Each entry has an id and a parent, so history forms
  a tree whose active branch supplies model context (pointer lines record the
  active leaf). Assistant messages record `api` (wire format) and `model`.
- **System prompt** — base instructions plus `AGENTS.md` files: the user's
  `~/.agents/AGENTS.md` and the nearest one discovered from the working
  directory upward. When both exist the prompt names both and their
  precedence, so the model reconciles user and project preferences.
- **Tools** — `read`, `grep`, `glob`, `bash`, `edit`, `write`, `subagent`, bound
  to the working directory. Tools run with the permissions of the tau process.
- **Search** — `grep` and `glob` walk the tree in Python (no ripgrep
  dependency), prune `.git`, `node_modules`, `.venv`, `__pycache__` and build
  output, skip binaries, and cap their output, so a broad query cannot flood
  the transcript. `grep` returns ripgrep-shaped `path:line: text` lines and
  supports `glob` filters, context lines, and content/count/file modes.
- **Sub-agents** — a `subagent` call runs a separate `AgentSession` with a fresh
  history and a reduced tool set: `read`, `grep`, `glob`, `bash`, with no
  `edit` or `write` (not read-only — `bash` can still write). Only its
  final report returns to the parent, so exploration does not accumulate in
  the main context. Sub-agents share the parent's backend (and therefore
  `/model`) and its abort flag. The terminal shows their tool calls live,
  with completion/error markers, without adding child output to the parent.
- **Compaction** — when the active branch passes a token threshold, the oldest
  turns are summarized into one message and the most recent turns are kept
  verbatim. Cuts land on user-message boundaries, so tool calls and their
  results are never split apart. Compaction only moves the active leaf; the
  original entries stay in the JSONL file.

## Long sessions

Compaction kicks in automatically, or on demand with `/compact`. The defaults
suit a ~200k-token context window:

| Variable | Default | Meaning |
| --- | --- | --- |
| `TAU_COMPACT` | `1` | Set to `0` to disable automatic compaction |
| `TAU_COMPACT_THRESHOLD` | `100000` | Estimated tokens in the branch that trigger it |
| `TAU_COMPACT_KEEP_TURNS` | `4` | Recent user turns kept verbatim |
| `TAU_COMPACT_INPUT_TOKENS` | `30000` | Cap per summarization request (longer histories are folded in batches) |

Lower the threshold when running a smaller-context model. Compaction is
conservative: if the summary would not be smaller than the turns it replaces
(short branches, or one huge tool result), it is skipped and the verbatim
history is kept. Failures are non-fatal too — the summary request fails,
history is left untouched, and the run continues.

## SDK usage

The SDK takes its configuration as arguments; only the `tau` command reads the
`TAU_*` environment variables.

```python
from tau import create_agent_session

def handler(event):
    print(event.get("delta", ""), end="")

session = create_agent_session(
    base_url="https://api.openai.com/v1",
    api_key="sk-",
)
unsubscribe = session.subscribe(handler)
session.prompt("What files are in the current directory?")
unsubscribe()

print(session.last_assistant_text())

session.dispose()
```

Custom tools implement the `Tool` interface:

```python
from tau.tools import Tool, ToolResult

class GitStatusTool(Tool):
    name = "git_status"
    description = "Show the working tree status."
    parameters = {"type": "object", "properties": {}}

    def execute(self, **_):
        import subprocess
        out = subprocess.run(["git", "status", "--short"], capture_output=True, text=True).stdout
        return ToolResult(output=out or "clean")
```
## Basic Safety Checks

The `policy.py` implements some basic gaurds for overtly destructive tool calls e.g. `rm -rf /`. These are not all encompassing and are not meant to catch every case, just the obvious ones.

## Layout

```
src/tau/
├── agent.py       # AgentSession: the agent loop
├── cli.py         # interactive REPL + print mode; reads TAU_* into the flags
├── compaction.py  # summarize old turns when the branch gets long
├── events.py      # session lifecycle events
├── messages.py    # message types (user, assistant, toolResult)
├── policy.py      # guard for tool calls (path escapes, obvious accidents)
├── prompts.py     # system prompt + AGENTS.md discovery
├── providers/     # model backends
│   ├── sse.py     # StreamReader: one POST, urllib + SSE, decoded into chunks
│   ├── backend.py # ModelBackend, both wire formats, and make_backend
│   ├── provider.py # Provider: model, endpoint, credentials, timeout
│   ├── errors.py  # ProviderError, and when a failure is worth retrying
├── render.py      # Markdown + unified-diff rendering
├── sandbox.py     # re-exec into a Docker container, cwd mounted at /workspace
├── session.py     # JSONL session persistence (tree-shaped)
├── spinner.py     # waiting-for-first-chunk spinner
├── theme.py       # ANSI styling, disabled for non-TTY / NO_COLOR
└── tools/         # read, grep, glob, bash, edit, write, subagent
```

