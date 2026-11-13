# lý

**Lý** is a minimal AI coding agent for your terminal. Sandboxed in Docker by default, written in a few thousand lines of dependency-free Python.

- **Small on purpose.** No frameworks, no dependencies. One process,
  standard library only.
- **Sandboxing built-in.** ly can run directly on your machine or as a sandboxed version of itself inside a docker container with your working directory automatically mounted in.
- **Just enough tools; no bloat.** `read`, `grep`, `glob`, `edit`, `write` and `bash`, plus a `subagent` that explores in its own context and reports back a summary.
- **Immutable history.** Conversation sessions are saved as jsonl files in `~/.ly/sessions/` . All history is append-only and immutable.

## Install

Requires Python 3.12+ (and Docker if you want to use sandboxing functionality) .


```bash
# with uv
uv tool install git+https://github.com/huyng/ly

# or with pip
python -m pip install git+https://github.com/huyng/ly
```

## Quick start

Specify the LLM model you want to use by setting `LY_MODEL` to `provider/model-name` and the api-key with `LY_API_KEY`. The `provider` tag lets ly determine the base_url required to reach the various llm model providers that ly has builtin support for.

```bash
export LY_MODEL=deepseek/deepseek-flash
export LY_API_KEY=sk-

ly "read the files in this directory and tell me what needs improvement"
```

Or keep the model in `~/.ly/settings.json` instead of your shell:

```json
{ "model": "deepseek/deepseek-flash" }
```


## Use

```bash
ly                                 # interactive session in this directory
ly -p "what does this repo do?"    # print mode: answer to stdout, nothing else
ly --resume 2                      # resume a recent session
ly --sandbox python:3.12           # run in another Docker image
ly -ns                             # run in non-sandboxed mode
```

Inside a session, you can access a variety of slash commands:

```
/model deepseek/deepseek-flash   switch model and provider mid-session
/help                            built-in and project commands
/compact                         summarize old turns now
/quit                            leave (also Ctrl-D)
```

A line starting with `!` runs a shell command in place.

```
!git status
!make test
```

Drop a Markdown file in `.agents/commands/` to add a command, so `review.md`
becomes `/review` and its text is the prompt. `$ARGUMENTS` becomes whatever you
type after the name.

## Configure


`~/.ly/settings.json` holds what you would otherwise set via environment variables:

```json
{
  "model": "deepseek/deepseek-flash",
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

## Agent SDK

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

Custom tools implement the `Tool` interface and go in the `tools` list. Passing
`tools` **replaces** the default coding tools; omit it to keep the defaults, or
include every desired tool in your list:

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
