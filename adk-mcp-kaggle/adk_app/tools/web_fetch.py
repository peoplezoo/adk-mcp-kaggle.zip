import httpx
from bs4 import BeautifulSoup

def web_fetch(url: str, timeout_sec: int = 20) -> dict:
    with httpx.Client(timeout=timeout_sec, follow_redirects=True) as s:
        r = s.get(url)
    ct = (r.headers.get("content-type") or "").lower()
    out = {"status": r.status_code, "headers": dict(r.headers), "url": str(r.url)}
    if "html" in ct:
        soup = BeautifulSoup(r.text, "lxml")
        for t in soup(["script","style","noscript"]): t.decompose()
        md = "\n".join(h.get_text(" ", strip=True) for h in soup.find_all(["h1","h2","h3","p","li"]))
        out.update({"title": soup.title.string if soup.title else None, "markdown": md})
    else:
        out.update({"bytes": len(r.content)})
    return out
