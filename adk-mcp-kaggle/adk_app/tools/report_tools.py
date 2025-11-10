import os, json

REPORTS = "/kaggle/working/agent/reports"
os.makedirs(REPORTS, exist_ok=True)

def report_md(title: str, notes: list[str], metrics_json: str, links: list[dict] | None = None) -> dict:
    try:
        with open(metrics_json,"r") as f: metrics = json.load(f)
    except Exception:
        metrics = {}
    body = [f"# {title}", "", "## Notes"] + [f"- {n}" for n in notes] + ["", "## Metrics", "```json", json.dumps(metrics, indent=2), "```"]
    if links:
        body += ["", "## Sources"] + [f"- [{x.get('title', x['url'])}]({x['url']})" for x in links]
    path = os.path.join(REPORTS, f"{title.lower().replace(' ','_')}.md")
    with open(path, "w", encoding="utf-8") as f: f.write("\n".join(body))
    return {"report_path": path}
