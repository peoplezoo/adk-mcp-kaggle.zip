"""FastAPI wrapper exposing ADK tool calls."""
from __future__ import annotations

import os
from typing import Any, Dict, Callable

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from adk_app.tools import get_adk_tools, load_allowlist


class LlmAgent:
    """Minimal agent façade that dispatches to registered tools."""

    def __init__(
        self,
        model: str,
        name: str,
        description: str,
        instruction: str,
        tools: Dict[str, Callable[..., Dict[str, Any]]],
    ) -> None:
        self.model = model
        self.name = name
        self.description = description
        self.instruction = instruction
        self.tools = tools

    def call_tool(self, name: str, **kwargs: Any) -> Dict[str, Any]:
        if name not in self.tools:
            raise KeyError(name)
        return self.tools[name](**kwargs)


allowlist = load_allowlist()
adk_tools = get_adk_tools(allowlist)

agent = LlmAgent(
    model=os.getenv("ADK_MODEL", "gemini-2.5-flash"),
    name="kaggle_assignment_agent",
    description="Research + ML baseline agent with shared tools",
    instruction=(
        "Use web_fetch for external resources, then dataset_load_csv, cv_split, tabular_baseline. "
        "Keep outputs compact; return JSON-friendly results."
    ),
    tools=adk_tools,
)

app = FastAPI()


class CallIn(BaseModel):
    tool: str
    args: Dict[str, Any] = {}


@app.post("/adk/call")
async def adk_call(inp: CallIn):
    try:
        out = agent.call_tool(inp.tool, **(inp.args or {}))
        return JSONResponse(out)
    except KeyError:
        return JSONResponse({"isError": True, "message": f"unknown tool {inp.tool}"}, status_code=404)
    except Exception as exc:  # pragma: no cover - safety net
        return JSONResponse({"isError": True, "message": str(exc)}, status_code=500)


@app.get("/health")
async def health():
    return {"status": "ok", "model": agent.model, "tools": sorted(agent.tools.keys())}
