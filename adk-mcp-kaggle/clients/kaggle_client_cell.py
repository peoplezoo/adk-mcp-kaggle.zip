# Kaggle client cell: call ADK and MCP endpoints from a notebook.
from __future__ import annotations

import json
import os

import httpx

ADK_BASE = os.environ.get("ADK_BASE", "https://<your-host>:8080")
MCP_BASE = os.environ.get("MCP_BASE", "https://<your-host>:8081")

with httpx.Client(timeout=20) as session:
    health = session.get(f"{ADK_BASE}/health").json()
print("ADK:", health)

# Example: inspect datasets available inside the Kaggle environment.
with httpx.Client(timeout=20) as session:
    datasets = session.post(f"{ADK_BASE}/adk/call", json={"tool": "dataset_list", "args": {}}).json()
print("Datasets:", json.dumps(datasets, indent=2)[:500])

# Example: load Titanic and run a baseline pipeline.
with httpx.Client(timeout=30) as session:
    load = session.post(
        f"{ADK_BASE}/adk/call",
        json={
            "tool": "dataset_load_csv",
            "args": {"dataset": "titanic", "filename": "train.csv"},
        },
    ).json()
    cv = session.post(
        f"{ADK_BASE}/adk/call",
        json={
            "tool": "cv_split",
            "args": {
                "cache_feather": load["cache_feather"],
                "target_col": "Survived",
                "n_splits": 5,
                "stratified": True,
            },
        },
    ).json()
    baseline = session.post(
        f"{ADK_BASE}/adk/call",
        json={
            "tool": "tabular_baseline",
            "args": {
                "cache_feather": load["cache_feather"],
                "cv_path": cv["cv_path"],
                "task": "classification",
            },
        },
    ).json()
print("Baseline:", json.dumps(baseline, indent=2))

# MCP: list tools and call web_fetch
with httpx.Client(timeout=20) as session:
    tools = session.get(f"{MCP_BASE}/mcp/tools").json()
print("MCP tools:", [tool["name"] for tool in tools.get("tools", [])])

with httpx.Client(timeout=20) as session:
    web = session.post(
        f"{MCP_BASE}/mcp/call",
        json={"name": "web_fetch", "arguments": {"url": "https://example.com"}},
    ).json()
print("web_fetch status:", web.get("status"))
