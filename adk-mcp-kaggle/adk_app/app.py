import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any

# ---- ADK shim (replace with google.adk imports when available) ----
class Tool:
    def __init__(self, fn): self.fn = fn

class LlmAgent:
    def __init__(self, model: str, name: str, description: str, instruction: str, tools: list):
        self.model = model
        self.name = name
        self.description = description
        self.instruction = instruction
        self.tools = {t.fn.__name__: t.fn for t in tools}
    def call_tool(self, name: str, **kwargs):
        return self.tools[name](**kwargs)

# ---- Tools ----
from adk_app.tools.web_fetch import web_fetch
from adk_app.tools.dataset_tools import dataset_list, dataset_load_csv
from adk_app.tools.cv_tools import cv_split
from adk_app.tools.baseline_tools import tabular_baseline
from adk_app.tools.report_tools import report_md

agent = LlmAgent(
    model=os.getenv("ADK_MODEL", "gemini-2.5-flash"),
    name="kaggle_assignment_agent",
    description="Research + ML baseline agent with shared tools",
    instruction=(
        "Use web_fetch for external resources, then dataset_load_csv, cv_split, tabular_baseline. "
        "Keep outputs compact; return JSON-friendly results."
    ),
    tools=[Tool(web_fetch), Tool(dataset_list), Tool(dataset_load_csv), Tool(cv_split), Tool(tabular_baseline), Tool(report_md)],
)

# ---- HTTP app ----
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
    except Exception as e:
        return JSONResponse({"isError": True, "message": str(e)}, status_code=500)

@app.get("/health")
async def health():
    return {"status": "ok", "model": agent.model, "tools": list(agent.tools.keys())}
