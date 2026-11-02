# τ tau

A minimal, extensible AI coding agent for the terminal, written in Python and
modeled after [pi](https://pi.dev).

tau replicates pi's core architecture — the agent loop, streaming events,
session persistence, tool execution, and context-file discovery — in roughly
1000 lines of dependency-free Python code (excluding tests).

## Installation & Usage

Requires Python 3.12+. There are no runtime dependencies — installation pulls
nothing beyond the standard library.

```bash
uv tool install git+https://github.com/huyng/tau
tau
```

Point tau to an LLM provider by setting environment variables. It is compatible with any OpenAI-API compatible endpoint.

```bash
export TAU_API_KEY=sk-              # API key for the selected provider
export TAU_MODEL=gpt-4.1            # The model you would like to use
export TAU_BASE_URL=
```

Only `TAU_API_KEY` is read for API keys; provider-specific key environment
variables are ignored. Use `--api-key` to override it for one invocation.

### Providers

**DeepSeek** — DeepSeek's API is OpenAI-compatible. Current model names are
`deepseek-flash` and `deepseek-v4-pro` (older deployments may still accept
`deepseek-chat` / `deepseek-reasoner`).

```bash
export TAU_BASE_URL=https://api.deepseek.com
export TAU_API_KEY=your-deepseek-key
export TAU_MODEL=deepseek-flash
```


**Z.ai (GLM models)** — [z.ai](https://z.ai) has two OpenAI-compatible
endpoints with **separate billing**. Pick the one matching where your
balance lives, or you will get "Insufficient balance" errors:

- **GLM Coding Plan subscription** → `https://api.z.ai/api/coding/paas/v4/`
- **Pay-as-you-go platform balance** → `https://api.z.ai/api/paas/v4/`

```bash
# GLM Coding Plan
export TAU_BASE_URL=https://api.z.ai/api/coding/paas/v4/
export TAU_API_KEY=your-zai-key
export TAU_MODEL=glm-4.6
```

```bash
# Pay-as-you-go
export TAU_BASE_URL=https://api.z.ai/api/paas/v4/
export TAU_API_KEY=your-zai-key
export TAU_MODEL=glm-4.6
```

**OpenAI Responses API** — from GPT-5.4 on, `chat/completions` allows function
tools only with `reasoning_effort: none`, and `gpt-6-astra` / `gpt-6.1-sol`
reject `none` outright. Those models work only on `/v1/responses`, which tau
speaks when asked (`--api responses` or `TAU_API=responses`):

```bash
export TAU_API=responses
export TAU_MODEL=gpt-6.1-sol
```

tau does not read `reasoning_content` off this endpoint; reasoning arrives as
a summary, so `/thinking` shows the model's own summary rather than raw
reasoning.


## Use

```bash
tau                                # interactive REPL in the current directory
tau "fix the failing tests"        # start with an initial prompt
tau -p "what does this repo do?"   # print mode: run once, write final response
tau --resume 2                     # resume a previous session
```

Ctrl-C aborts the active run (steering between turns). Ctrl-D exits.

## How it works

A session is tau's record of a conversation. The pieces mirror pi:

- **Agent loop** — a submitted message is added to the active branch; tau
  builds a model request from the system prompt, active branch, and tool
  definitions; the provider streams an assistant response containing text
  and/or tool calls; tau executes each tool call and records the results.
  Tool results trigger another turn; otherwise the run ends.
- **Provider** — one request per turn — `chat/completions` by default, or
  `/v1/responses` for models that refuse tools on chat-completions — streamed over
  `urllib` and parsed line by line as SSE, so there is no client SDK to depend
  on. Text, reasoning, and tool-call deltas accumulate into the finished
  message; `transport=` swaps the HTTP layer out (the tests inject a stand-in).
  Transient failures (429, 5xx, dropped streams, server overload) are retried up
  to three times with 1/2/4s backoff; permanent ones (bad key, bad request,
  content filtered) fail immediately. Ctrl-C aborts between chunks.
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
- **Tools** — `read`, `grep`, `glob`, `bash`, `edit`, `write`, `task`, bound
  to the working directory. Tools run with the permissions of the tau process.
- **Search** — `grep` and `glob` walk the tree in Python (no ripgrep
  dependency), prune `.git`, `node_modules`, `.venv`, `__pycache__` and build
  output, skip binaries, and cap their output, so a broad query cannot flood
  the transcript. `grep` returns ripgrep-shaped `path:line: text` lines and
  supports `glob` filters, context lines, and content/count/file modes.
- **Sub-agents** — a `task` call runs a separate `AgentSession` with a fresh
  history and a reduced tool set: `read`, `grep`, `glob`, `bash`, with no
  `edit` or `write` (not read-only — `bash` can still write). Only its
  final report returns to the parent, so exploration does not accumulate in
  the main context. Sub-agents share the parent's provider (and therefore
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
│   ├── sse.py     # one POST, urllib + SSE, decoded into JSON chunks
│   ├── endpoint.py # shared interface and endpoint configuration
│   ├── errors.py  # ProviderError, and when a failure is worth retrying
│   ├── chat.py    # /chat/completions backend
│   └── responses.py # /v1/responses backend
├── render.py      # Markdown + unified-diff rendering
├── session.py     # JSONL session persistence (tree-shaped)
├── spinner.py     # waiting-for-first-chunk spinner
├── theme.py       # ANSI styling, disabled for non-TTY / NO_COLOR
└── tools/         # read, grep, glob, bash, edit, write, task
```

