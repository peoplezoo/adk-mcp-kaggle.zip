# Kaggle client cell: call ADK and MCP endpoints from a notebook.
import os, httpx, json

ADK_BASE = os.environ.get("ADK_BASE", "https://<your-host>:8080")
MCP_BASE = os.environ.get("MCP_BASE", "https://<your-host>:8081")

with httpx.Client(timeout=20) as s:
    h = s.get(f"{ADK_BASE}/health").json()
print("ADK:", h)

# Example: load Titanic and run baseline
with httpx.Client(timeout=30) as s:
    load = s.post(f"{ADK_BASE}/adk/call", json={"tool":"dataset_load_csv","args":{"dataset":"titanic","filename":"train.csv"}}).json()
    cv   = s.post(f"{ADK_BASE}/adk/call", json={"tool":"cv_split","args":{"cache_feather":load["cache_feather"],"target_col":"Survived","n_splits":5,"stratified":True}}).json()
    base = s.post(f"{ADK_BASE}/adk/call", json={"tool":"tabular_baseline","args":{"cache_feather":load["cache_feather"],"cv_path":cv["cv_path"],"task":"classification"}}).json()
print("Baseline:", base)

# MCP: list tools and call web_fetch
with httpx.Client(timeout=20) as s:
    tools = s.get(f"{MCP_BASE}/mcp/tools").json()
print("MCP tools:", [t["name"] for t in tools.get("tools",[])])

with httpx.Client(timeout=20) as s:
    wf = s.post(f"{MCP_BASE}/mcp/call", json={"name":"web_fetch","arguments":{"url":"https://example.com"}}).json()
print("web_fetch status:", wf.get("status"))
