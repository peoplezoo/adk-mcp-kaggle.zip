from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
import json, importlib
from pathlib import Path

app = FastAPI()
SCHEMAS = Path(__file__).parent / "schemas"

# Map MCP tool names → ADK tool implementations
TOOL_IMPLS = {
    "web_fetch": ("adk_app.tools.web_fetch", "web_fetch"),
    "dataset.load_csv": ("adk_app.tools.dataset_tools", "dataset_load_csv"),
    "cv.split": ("adk_app.tools.cv_tools", "cv_split"),
    "tabular.baseline": ("adk_app.tools.baseline_tools", "tabular_baseline"),
    "report.md": ("adk_app.tools.report_tools", "report_md"),
}

class MCPCall(BaseModel):
    name: str
    arguments: Dict[str, Any] = {}

@app.get("/mcp/tools")
def list_tools():
    tools = []
    for p in SCHEMAS.glob("*.json"):
        tools.append(json.loads(p.read_text()))
    return {"tools": tools}

@app.post("/mcp/call")
def call_tool(inp: MCPCall):
    if inp.name not in TOOL_IMPLS:
        return {"isError": True, "message": f"unknown tool {inp.name}"}
    mod_name, fn_name = TOOL_IMPLS[inp.name]
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name)
    try:
        return fn(**(inp.arguments or {}))
    except Exception as e:
        return {"isError": True, "message": str(e)}
