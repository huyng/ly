# lý

**Lý** is a minimal AI coding agent for your terminal. Sandboxed in Docker by default,
written in a few thousand lines of dependency-free Python.

- **Small on purpose.** No frameworks, no dependencies. One process,
  standard library only.
- **Sandboxed by default.** ly transparently runs in a container with your project mounted, so it can only touch the project.
- **Useful tools.** `read`, `grep`, `glob`, `edit`, `write` and `bash`, plus a
  `subagent` that explores in its own context and reports back a summary.
- **Immutable history** Conversation sessions are saved files
- **Your endpoint.** OpenAI and DeepSeek are built in; any OpenAI-compatible
  gateway is one entry in the settings file.

## Install

Requires Python 3.12+. Nothing is pulled in but the standard library.

```bash
uv tool install git+https://github.com/huyng/ly
```

## Quick start

A model is always written `provider/model`. The provider half names a preset
that knows the endpoint and where its key lives, so the only thing you supply
is the key. There is no default model: say which one you want once.

```bash
export LY_MODEL=deepseek/deepseek-chat
export DEEPSEEK_API_KEY=sk-       # or LY_API_KEY for either provider

ly "fix the failing tests"
```

Or keep the model in `~/.ly/settings.json` instead of your shell:

```json
{ "model": "deepseek/deepseek-chat" }
```

Without a model, ly stops at startup and says how to set one, rather than
picking an endpoint you did not ask for.

## Use

```bash
ly                                 # interactive session in this directory
ly -p "what does this repo do?"    # print mode: answer to stdout, nothing else
ly --resume 2                      # resume a recent session
ly --sandbox python:3.12           # run in another Docker image
ly --nosandbox                     #
```

Inside a session, Tab completes commands and model names:

```
/model deepseek/deepseek-chat   switch model and provider mid-session
/help                           built-in and project commands
/compact                        summarize old turns now
/quit                           leave (also Ctrl-D)
```

A line starting with `!` runs a shell command in place and leaves the
conversation alone — the model never sees the command or its output:

```
!git status
!make test
```

Drop a Markdown file in `.agents/commands/` to add a command, so `review.md`
becomes `/review` and its text is the prompt. `$ARGUMENTS` becomes whatever you
type after the name.

## Configure

`~/.ly/settings.json` holds what you would otherwise export, and an `LY_*`
variable beats it for one run:

```json
{
  "model": "deepseek/deepseek-chat",
  "sandbox": true,
  "thinking": true
}
```

| Variable | Meaning |
| --- | --- |
| `LY_MODEL` | Model to use, always `provider/model` (required) |
| `LY_API_KEY` | Key for the provider, unless its own variable is set |
| `LY_SANDBOX` | Set to `0` to run on the host instead of inside Docker sandbox |
| `LY_SANDBOX_IMAGE` | Image to run in (default `python:3.12`) |
| `LY_SESSIONS_DIR` | Where sessions are written (default `~/.ly/sessions`) |

Run `/config` to see what is actually in effect. Other OpenAI-compatible
endpoints, the remaining variables and the compaction knobs are covered in
[DEVELOPMENT.md](DEVELOPMENT.md).

## Sandboxing

By default **ly** re-executes itself as `docker run`, mounting the working
directory at `/workspace`. The container runs as you, so files it writes belong
to you, and the banner names the image and the mount:

```
  • sandboxed docker python:3.12 · /Users/you/ly → /workspace
```

The default image carries git, make, gcc and curl. Point `--sandbox` at
another image when you need more; it must have Python 3.12 or newer. podman
works too, including rootless podman behind the `docker` shim.

## Embedding ly

ly is also a library. `create_agent_session` takes its configuration as
arguments — only the `ly` command reads `LY_*` variables — so the agent loop
can live inside your own program.

```python
from ly import create_agent_session


def handler(event):
    # Events include text deltas, tool calls, and run lifecycle.
    print(event.get("delta", ""), end="")


session = create_agent_session(model="openai/gpt-4.1", api_key="sk-")
unsubscribe = session.subscribe(handler)
session.prompt("What files are in the current directory?")
unsubscribe()

print(session.last_assistant_text())
session.dispose()
```

Custom tools implement the `Tool` interface and go in the `tools` list:

```python
from ly.tools import Tool, ToolResult


class GitStatusTool(Tool):
    name = "git_status"
    description = "Show the working tree status."
    parameters = {"type": "object", "properties": {}}

    def execute(self, **_):
        import subprocess

        out = subprocess.run(
            ["git", "status", "--short"], capture_output=True, text=True
        ).stdout
        return ToolResult(output=out or "clean")


session = create_agent_session(
    model="openai/gpt-4.1", api_key="sk-", tools=[GitStatusTool()]
)
```
