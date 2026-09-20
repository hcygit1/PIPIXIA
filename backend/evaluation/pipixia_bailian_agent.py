"""Minimal OpenAI-compatible ReAct agent used inside SkillLearnBench containers."""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
from pathlib import Path
from typing import Any


MAX_TOOL_OUTPUT = 16_000
MAX_MODEL_OUTPUT = 2_048
SKILL_ROOTS = (
    Path("/root/.agents/skills"),
    Path("/root/.codex/skills"),
    Path("/root/.claude/skills"),
)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a bash command in the isolated task container.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 300},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file with optional line slicing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer", "minimum": 1},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 2000},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a UTF-8 text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files matching a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                },
                "required": ["pattern"],
            },
        },
    },
]


def _completion_options(model: str) -> dict[str, Any]:
    """Keep benchmark turns bounded; GLM reasoning is unnecessary for tool execution."""
    options: dict[str, Any] = {"max_tokens": MAX_MODEL_OUTPUT}
    if model.lower().startswith("glm-"):
        options["extra_body"] = {"enable_thinking": False}
    return options


def _read_instruction(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig").strip()
    except UnicodeDecodeError:
        return path.read_text(encoding="gb18030").strip()


def _truncate(value: str) -> str:
    if len(value) <= MAX_TOOL_OUTPUT:
        return value
    half = MAX_TOOL_OUTPUT // 2
    return value[:half] + "\n...[tool output truncated]...\n" + value[-half:]


def load_skills(roots: tuple[Path, ...] = SKILL_ROOTS) -> list[tuple[str, str]]:
    """Load one physical skill tree once even when the benchmark copies it to many roots."""
    seen: set[str] = set()
    skills: list[tuple[str, str]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("SKILL.md")):
            try:
                content = path.read_text(encoding="utf-8")
            except OSError:
                continue
            fingerprint = content.strip()
            if not fingerprint or fingerprint in seen:
                continue
            seen.add(fingerprint)
            skills.append((str(path), content))
    return skills


def build_system_prompt(skills: list[tuple[str, str]]) -> str:
    prompt = (
        "You are PIPIXIA running inside an isolated benchmark container. "
        "Complete the user's task by inspecting and modifying the real container filesystem. "
        "Use tools until the requested artifact is correct. Do not merely explain what should be done. "
        "You may run verification commands, but do not read /tests or verifier files."
    )
    if not skills:
        return prompt
    rendered = "\n\n".join(
        f"<skill path={json.dumps(path)}>\n{content}\n</skill>"
        for path, content in skills
    )
    return f"{prompt}\n\nThe following skills are available. Apply them when relevant:\n\n{rendered}"


def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    try:
        if name == "shell":
            timeout = min(max(int(arguments.get("timeout_seconds", 120)), 1), 300)
            completed = subprocess.run(
                str(arguments["command"]),
                shell=True,
                executable="/bin/bash",
                cwd="/root",
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = (
                f"exit_code={completed.returncode}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
            return _truncate(output)
        if name == "read_file":
            path = Path(str(arguments["path"])).expanduser()
            if not path.is_absolute():
                path = Path("/root") / path
            lines = path.read_text(encoding="utf-8").splitlines()
            start = max(int(arguments.get("offset", 1)) - 1, 0)
            limit = int(arguments.get("limit", 500))
            return _truncate("\n".join(lines[start:start + limit]))
        if name == "write_file":
            path = Path(str(arguments["path"])).expanduser()
            if not path.is_absolute():
                path = Path("/root") / path
            path.parent.mkdir(parents=True, exist_ok=True)
            content = str(arguments["content"])
            path.write_text(content, encoding="utf-8")
            return f"wrote {len(content)} characters to {path}"
        if name == "list_files":
            pattern = str(arguments["pattern"])
            if not Path(pattern).is_absolute():
                pattern = str(Path("/root") / pattern)
            return _truncate("\n".join(sorted(glob.glob(pattern, recursive=True))[:2000]))
        return f"unknown tool: {name}"
    except subprocess.TimeoutExpired:
        return "tool error: command timed out"
    except Exception as exc:
        return f"tool error: {type(exc).__name__}: {exc}"


def _parse_arguments(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


def run(instruction: str, *, model: str, max_steps: int) -> int:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["PIPIXIA_LLM_BASE_URL"],
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt(load_skills())},
        {"role": "user", "content": instruction},
    ]
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    for step in range(1, max_steps + 1):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            **_completion_options(model),
        )
        if response.usage:
            usage["input_tokens"] += int(response.usage.prompt_tokens or 0)
            usage["output_tokens"] += int(response.usage.completion_tokens or 0)
            usage["total_tokens"] += int(response.usage.total_tokens or 0)

        message = response.choices[0].message
        assistant_entry: dict[str, Any] = {
            "role": "assistant",
            "content": message.content or "",
        }
        if message.tool_calls:
            assistant_entry["tool_calls"] = [call.model_dump() for call in message.tool_calls]
        messages.append(assistant_entry)
        print(json.dumps({"type": "message", **assistant_entry}, ensure_ascii=False), flush=True)

        if not message.tool_calls:
            print(json.dumps({"type": "result", "usage": usage, "steps": step}), flush=True)
            return 0

        for call in message.tool_calls:
            arguments = _parse_arguments(call.function.arguments)
            output = execute_tool(call.function.name, arguments)
            tool_entry = {
                "role": "tool",
                "tool_call_id": call.id,
                "content": output,
            }
            messages.append(tool_entry)
            print(
                json.dumps(
                    {
                        "type": "tool_result",
                        "tool": call.function.name,
                        "arguments": arguments,
                        **tool_entry,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    print(json.dumps({"type": "result", "usage": usage, "steps": max_steps, "error": "max_steps"}), flush=True)
    return 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction-file", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    args = parser.parse_args()
    instruction = _read_instruction(args.instruction_file)
    raise SystemExit(run(instruction, model=args.model, max_steps=args.max_steps))


if __name__ == "__main__":
    main()
