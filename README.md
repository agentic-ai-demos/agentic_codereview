---

title: Code Review Agent based on MCP
emoji: 🛰️
colorFrom: indigo
colorTo: blue
sdk: docker
app_file: Dockerfile
pinned: false
---

# Agentic AI Code Review in Pure Python (MCP-powered)

Agentic code review pipeline that plans, calls tools, and produces structured findings without any heavyweight framework. Implemented in plain Python using [uv](https://github.com/astral-sh/uv), [FastAPI](https://fastapi.tiangolo.com/), and [nmagents](https://github.com/alexcpn/noagent-ai). Deep code context comes from a Tree-Sitter-backed Model Context Protocol (MCP) server.

## Why this repo is interesting
- End-to-end AI review loop in a few hundred lines of Python (`code_review_agent.py`)
- Tool-augmented LLM via Tree-Sitter AST introspection from an MCP server
- Deterministic step planning/execution with JSON repair and YAML logs
- Works with OpenAI or any OpenAI-compatible endpoint (ollam,vllm)
- Ships as a FastAPI service, CLI helper, and Docker image

## How it works
- Fetch the PR diff, ask the LLM for a per-file review plan, then execute each step.
- MCP server ([codereview_mcp_server](https://github.com/alexcpn/codereview_mcp_server)) exposes AST tools (definitions, call-sites, docstrings) using [Tree-Sitter](https://tree-sitter.github.io/tree-sitter/).
- Minimal orchestration comes from [nmagents](https://github.com/alexcpn/noagent-ai) Command pattern: plan → optional tool calls → critique/patch suggestions → YAML logs.

### Core flow (excerpt from `code_review_agent.py`)
```python
file_diffs = git_utils.get_pr_diff_url(repo_url, pr_number)
response = call_llm_command.execute(context)                # plan steps
response_data, _ = parse_json_response_with_repair(...)     # repair/parse plan

tools = step.get("tools", [])
if tools:
    tool_outputs = await execute_step_tools(step, ast_tool_call_command)

step_context = load_prompt(diff_or_code_block=diff, tool_outputs=step.get("tool_results", ""))
step_response = call_llm_command.execute(step_context)      # execute each step
```

## Prerequisites
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) installed
- `.env` with `OPENAI_API_KEY=...`
- Running MCP server with AST tools (e.g., [codereview_mcp_server](https://github.com/alexcpn/codereview_mcp_server)) reachable at `CODE_AST_MCP_SERVER_URL`

## Setup
```bash
uv sync  # install dependencies from pyproject.toml
# create .env with OPENAI_API_KEY and optionally CODE_AST_MCP_SERVER_URL
```

## Start the Tree-Sitter MCP server
```bash
git clone https://github.com/alexcpn/codereview_mcp_server.git
cd codereview_mcp_server
uv run python http_server.py  # serves MCP at http://127.0.0.1:7860/mcp/
```

## Run the agent API
```bash
CODE_AST_MCP_SERVER_URL=http://127.0.0.1:7860/mcp/ \
uv run uvicorn code_review_agent:app --host 0.0.0.0 --port 8860
```

## Trigger a review
- CLI helper (default GET /review):
```bash
uv run python client.py --repo-url https://github.com/huggingface/accelerate --pr-number 3321
```
- Curl:
```bash
curl "http://127.0.0.1:8860/review?repo_url=https://github.com/huggingface/accelerate&pr_number=3321"
```
- Optional webhook-style POST: `python client.py --use-webhook ...` (see `client.py` for payload shape).

Logs land in `logs/` with per-step YAML outputs. See `sample_logs/` for a captured run.

## Sample artifacts
- Plan produced by the LLM: `sample_logs/step_1_20251201103933.yaml`
- Tool result snippet from MCP: `sample_logs/out_20251201103933_.log`
- Structured findings per step: `sample_logs/step_1_step1_done_20251201103933.yaml`, `sample_logs/step_2_step1_done_20251201103933.yaml`

Example finding (truncated):
```yaml
title: Clarify 'reserve_max_layer' impact on device allocation
severity: medium
file: tests/test_modeling_utils.py
why_it_matters: Tests assume reserve_max_layer preserves capacity; mismatch risks false failures.
fix:
  strategy: Update tests to explicitly specify the intended behavior.
```

## Docker
```bash
docker build -t codereview-agent .
docker run -it --rm -p 7860:7860 codereview-agent
```

## References
- [Model Context Protocol](https://github.com/modelcontextprotocol/specification)
- [Tree-Sitter](https://tree-sitter.github.io/tree-sitter/)
- [codereview_mcp_server](https://github.com/alexcpn/codereview_mcp_server)
- [nmagents (noagent-ai)](https://github.com/alexcpn/noagent-ai)
- [uv package manager](https://github.com/astral-sh/uv)
