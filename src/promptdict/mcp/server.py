"""Stdio MCP for PromptDictCompress."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from promptdict.mcp.protocol import run_mcp_loop

TOOLS = [
    {
        "name": "compress_text",
        "description": "Lossless dictionary-encode text into a packed prompt form.",
        "inputSchema": {
            "type": "object",
            "required": ["text"],
            "properties": {
                "text": {"type": "string"},
                "min_freq": {"type": "integer", "default": 3},
                "max_dict_size": {"type": "integer", "default": 256},
            },
        },
    },
    {
        "name": "compress_hierarchical",
        "description": "PageIndex hierarchical compress of text.",
        "inputSchema": {
            "type": "object",
            "required": ["text"],
            "properties": {
                "text": {"type": "string"},
                "page_size": {"type": "integer", "default": 4000},
            },
        },
    },
]


def _result_obj(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        d = dict(obj.__dict__)
        # common fields
        for k in ("prompt", "packed", "text", "ratio", "dictionary"):
            if hasattr(obj, k) and k not in d:
                d[k] = getattr(obj, k)
        return {k: (v if isinstance(v, (str, int, float, bool, type(None), list, dict)) else str(v)) for k, v in d.items()}
    return {"value": str(obj)}


def dispatch(name: str, args: dict[str, Any]) -> dict[str, Any]:
    from promptdict.compress import compress_hierarchical, compress_text

    if name == "compress_text":
        r = compress_text(
            str(args.get("text") or ""),
            min_freq=int(args.get("min_freq") or 3),
            max_dict_size=int(args.get("max_dict_size") or 256),
        )
        return _result_obj(r)
    if name == "compress_hierarchical":
        r = compress_hierarchical(
            str(args.get("text") or ""),
            page_size=int(args.get("page_size") or 4000),
        )
        return _result_obj(r)
    return {"error": f"unknown tool: {name}"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="PromptDict MCP")
    p.add_argument("--jsonl", action="store_true")
    args = p.parse_args(argv)
    if args.jsonl:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            method = msg.get("method")
            if method in {"tools/list", "list_tools"}:
                print(json.dumps({"tools": TOOLS}), flush=True)
            elif method in {"tools/call", "call"}:
                name = msg.get("name") or (msg.get("params") or {}).get("name")
                a = msg.get("arguments") or (msg.get("params") or {}).get("arguments") or {}
                print(json.dumps({"result": dispatch(str(name), dict(a))}, default=str), flush=True)
            else:
                print(json.dumps({"error": method}), flush=True)
        return 0
    return run_mcp_loop(
        server_name="promptdict",
        server_version="0.3.0",
        list_tools=lambda: TOOLS,
        call_tool=dispatch,
    )


if __name__ == "__main__":
    raise SystemExit(main())
