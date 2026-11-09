# lý

A minimal, extensible AI coding agent for the terminal, written in Python and modeled after [pi](https://pi.dev). It transparently runs sandboxed inside Docker by default.

When you run `ly` it automatically forks and launches itself inside a docker container mounted with your current working directory.

ly replicates pi's core architecture — the agent loop, streaming events,
session persistence, tool execution, and context-file discovery — in roughly
3000 lines of dependency-free Python code.

## Installation & Usage

Requires Python 3.12+. There are no runtime dependencies — installation pulls nothing beyond the standard library.

```bash
uv tool install git+https://github.com/huyng/ly
```

Point ly to an LLM provider by setting environment variables. It is compatible with any OpenAI-API compatible endpoint.

```bash
export LY_API_KEY=sk-             # API key for the selected provider
export LY_MODEL=gpt-4.1           # The model you would like to use
export LY_BASE_URL=
```

### Environment variables

Every setting can come from the environment; a command-line flag wins where
one exists. These are read by `ly` itself:

| Variable | Default | Meaning |
| --- | --- | --- |
| `LY_API_KEY` | — | API key for the provider (required) |
| `LY_BASE_URL` | — | OpenAI-compatible base URL, such as `https://api.openai.com/v1` (required) |
| `LY_MODEL` | `gpt-4.1` | Model to ask for |
| `LY_API` | `chat` | Wire format: `chat` for `/chat/completions`, `responses` for `/responses` |
| `LY_PASS_REASONING_CONTENT` | on when the base URL contains `deepseek` | Echo `reasoning_content` back on tool calls, which DeepSeek requires. `0`, `false`, `no` or `off` turns it off |
| `LY_DOCKER` | `1` | Set to `0` to run on the host instead of in Docker (same as `--nosandbox`) |
| `LY_DOCKER_IMAGE` | `python:3.12` | Image to run in when sandboxed (same as `--sandbox`) |
| `LY_SESSIONS_DIR` | `~/.ly/sessions` | Where session JSONL files are written |
| `LY_COMPACT` | `1` | Set to `0` to disable automatic compaction |
| `LY_COMPACT_THRESHOLD` | `100000` | Estimated tokens in the branch that trigger compaction |
| `LY_COMPACT_KEEP_TURNS` | `4` | Recent user turns kept verbatim |
| `LY_COMPACT_INPUT_TOKENS` | `30000` | Cap per summarization request; longer histories are folded in batches |

Styling also follows the usual terminal variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `NO_COLOR` | unset | Any value disables ANSI styling |
| `FORCE_COLOR` | unset | Any value forces styling, unless `NO_COLOR` is also set |
| `TERM` | from the shell | `dumb` disables styling |

`LY_SB_*` variables are set by the host for the copy of ly inside the
container and are not meant to be set by hand; see
[Sandboxing with Docker](#sandboxing-with-docker).

### Settings file

Anything you would otherwise export can live in `~/.ly/settings.json`
instead. Every key is optional, and a missing or malformed file simply means
defaults, so a typo cannot stop `ly` from starting.

```json
{
  "model": "gpt-4.1",
  "base_url": "https://api.openai.com/v1",
  "api": "chat",
  "markdown": true,
  "thinking": true,
  "sandbox": true,
  "compaction": { "threshold_tokens": 32000 }
}
```

An `LY_*` variable beats this file, which beats the built-in default:
`LY_MODEL=deepseek-chat ly` overrides `"model"` here for one run. The
environment is read in one place, `ly/settings.py`, which is also where the
file is read, so there is a single answer to "where did this value come
from?". `sandbox: false` is the same as `LY_DOCKER=0` or passing
`--nosandbox` every time, `markdown` and `thinking` set the starting state of
`/plain` and `/thinking`, and `api` must be `chat` or `responses`. The
provider options have no command-line flags: they come from `LY_*` and this
file only, so the host and the sandbox cannot disagree about which endpoint
they are talking to. Unknown keys and values of the wrong type are ignored
one at a time. The API key is not read from this file; keep it in
`LY_API_KEY`. The Docker image is not read from it either: use `--sandbox`
or `LY_DOCKER_IMAGE`. Run `/config` to see the values actually in effect.

## Use

```bash
ly                                # interactive REPL in the current directory
ly "fix the failing tests"        # start with an initial prompt
ly -p "what does this repo do?"   # print mode: run once, write final response
ly --resume 2                     # resume a previous session
ly -s python:3.12                 # run in the named Docker image
```

### Custom slash commands

Drop a Markdown file in `~/.agents/commands/` to define a command for every
project, or in this directory's `.agents/commands/` to define one for a single
project. Ancestors are not searched, so a command comes from one of those two
places and nowhere else. The file name is the command name, so `review.md`
becomes `/review`, and the file's text is the prompt ly sends:

```markdown
# Review the pending diff

Review the uncommitted changes with `git diff`. Comment on correctness and
style. $ARGUMENTS
```

```bash
ly> /review be brief
```

`$ARGUMENTS` is replaced by the text typed after the command name; without
that placeholder the arguments are appended as a final paragraph. Project
files win over user files with the same name, and the built-in commands above
always win. `/help` lists what was found, and files are read when ly starts,
so restart it after adding one.

### Sandboxing with Docker

By default, `ly` runs inside a Docker container by forking and re-executing itself as `docker run`, mounting the working directory at `/workspace`, so the tools can only touch the project. Sessions are bind-mounted from the host's `~/.ly/sessions`, and `~/.agents` is mounted read-only.

The container runs as your host uid/gid, so files it writes into the project
belong to you. Its `HOME` moves to `/tmp`, because a non-root uid cannot write
the image's `/root`; anything the `bash` tool sets up via `~` (a `~/.gitconfig`, say) therefore lasts only for that container.

podman works too, including rootless podman behind the `docker` shim.

Variables starting with `LY_SB_` describe the container to the copy of ly
running inside it: `LY_SB_IN_CONTAINER`, `LY_SB_IMAGE`, `LY_SB_HOST_DIR` and
`LY_SB_SESSIONS_DIR`. They are set by the host for that run and never forwarded,
so a stale value cannot describe the wrong container. `LY_DOCKER` and
`LY_DOCKER_IMAGE` also stay behind, because they chose what to start. Every
other `LY_*` variable is a setting and travels into the container unchanged,
which is how the host and the container always agree on where to point and how
to behave.

```bash
ly -s python:3.12                        # image to use (default python:3.12)
LY_DOCKER_IMAGE=python:3.12 ly           # the same choice from the environment
```

The flag wins over the variable. The default image is `python:3.12`, which
carries git, make, gcc and curl, so the `bash` tool has the usual build tools.
Point `--sandbox` or `LY_DOCKER_IMAGE` at another image when you need more; it
must provide `python3` (or `python`) at version 3.12 or newer, since that is
what ly runs on inside it. The startup banner names the image in use and both
directories, so the container is never a mystery and the host path is there to
copy into another shell:

```
  • sandboxed docker python:3.12 · /Users/you/ly → /workspace
```

In print mode the same fact goes to stderr (`ly: docker python:3.12`) and
stdout carries only the answer, so `ly -p … | jq` keeps working.

You can run ly without a sandbox with the `--nosandbox` flag.

```bash
ly --nosandbox     # stay on the host (or LY_DOCKER=0)
```

## How it works

A session is ly's record of a conversation. The pieces mirror pi:

- **Agent loop** — a submitted message is added to the active branch; ly
  builds a model request from the system prompt, active branch, and tool
  definitions; the provider streams an assistant response containing text
  and/or tool calls; ly executes each tool call and records the results.
  Tool results trigger another turn; otherwise the run ends.
- **Backend** — one request per turn — `chat/completions` by default, or
  `/v1/responses` for models that refuse tools on chat-completions — streamed over
  `urllib` and parsed line by line as SSE, so there is no client SDK to depend
  on. The backend holds a `Provider` (model, endpoint, credentials) and reads it
  on every request, so a model switch takes effect immediately. Text, reasoning,
  and tool-call deltas accumulate into the finished message.
- **Events** — subscribe with `session.subscribe(callback)` to observe
  message start/end, text deltas, tool execution, and run lifecycle.
- **Sessions** — append-only JSONL under `~/.ly/sessions/<date>/` (override
  with `LY_SESSIONS_DIR`). Each entry has an id and a parent, so history forms
  a tree whose active branch supplies model context (pointer lines record the
  active leaf). Assistant messages record `api` (wire format) and `model`.
- **System prompt** — base instructions plus `AGENTS.md` files: the user's
  `~/.agents/AGENTS.md` and the one in the working directory. Ancestors are
  not searched, so run ly from the project root to pick up its instructions.
  When both exist the prompt names both and their precedence, so the model
  reconciles user and project preferences.
- **Tools** — `read`, `grep`, `glob`, `bash`, `edit`, `write`, `subagent`, bound
  to the working directory. Tools run with the permissions of the ly process.
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

Compaction kicks in automatically, or on demand with `/compact`. Its defaults
(`LY_COMPACT`, `LY_COMPACT_THRESHOLD`, `LY_COMPACT_KEEP_TURNS` and
`LY_COMPACT_INPUT_TOKENS`, listed above) suit a ~200k-token context window.

All four can go in the settings file under a `compaction` object instead,
where a variable still wins:

```json
{
  "compaction": {
    "threshold_tokens": 32000,
    "keep_turns": 2
  }
}
```

Lower the threshold when running a smaller-context model. Compaction is
conservative: if the summary would not be smaller than the turns it replaces
(short branches, or one huge tool result), it is skipped and the verbatim
history is kept. Failures are non-fatal too — the summary request fails,
history is left untouched, and the run continues.

## SDK usage

The SDK takes its configuration as arguments; only the `ly` command reads the
`LY_*` environment variables.

```python
from ly import create_agent_session

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
from ly.tools import Tool, ToolResult

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
src/ly/
├── agent.py       # AgentSession: the agent loop
├── cli.py         # interactive REPL + print mode; reads LY_* into the flags
├── commands.py    # Markdown-defined slash commands (.agents/commands)
├── compaction.py  # summarize old turns when the branch gets long
├── events.py      # session lifecycle events
├── messages.py    # message types (user, assistant, toolResult)
├── policy.py      # guard for tool calls (path escapes, obvious accidents)
├── prompts.py     # system prompt + AGENTS.md discovery (cwd and ~/.agents)
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

