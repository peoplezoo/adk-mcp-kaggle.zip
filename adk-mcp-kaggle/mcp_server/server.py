"""FastAPI server exposing the tool registry over MCP."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel

from adk_app.tools import get_mcp_tools, load_allowlist

app = FastAPI()
SCHEMAS = Path(__file__).parent / "schemas"

allowlist = load_allowlist()
MCP_TOOLS = get_mcp_tools(allowlist)


class MCPCall(BaseModel):
    name: str
    arguments: Dict[str, Any] = {}


@app.get("/mcp/tools")
def list_tools():
    tools = []
    for schema_file in SCHEMAS.glob("*.json"):
        schema = json.loads(schema_file.read_text())
        if schema.get("name") in MCP_TOOLS:
            tools.append(schema)
    return {"tools": tools}


@app.post("/mcp/call")
def call_tool(inp: MCPCall):
    if inp.name not in MCP_TOOLS:
        return {"isError": True, "message": f"unknown tool {inp.name}"}
    tool_fn = MCP_TOOLS[inp.name]
    try:
        return tool_fn(**(inp.arguments or {}))
    except Exception as exc:  # pragma: no cover - safety net
        return {"isError": True, "message": str(exc)}
