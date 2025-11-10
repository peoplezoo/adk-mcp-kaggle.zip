import os, pandas as pd

BASE = "/kaggle/input"

def dataset_list() -> dict:
    items = []
    if os.path.isdir(BASE):
        for x in sorted(os.listdir(BASE)):
            p = os.path.join(BASE, x)
            if os.path.isdir(p): items.append({"name": x, "path": p})
    return {"datasets": items}

def dataset_load_csv(dataset: str, filename: str, nrows: int | None = None) -> dict:
    path = os.path.join(BASE, dataset, filename)
    if not os.path.exists(path):
        return {"isError": True, "message": f"not found: {path}"}
    df = pd.read_csv(path, nrows=nrows)
    meta = {"rows": int(df.shape[0]), "cols": int(df.shape[1]), "columns": df.columns.tolist()}
    cache = "/kaggle/working/agent/cache"
    os.makedirs(cache, exist_ok=True)
    feather_path = os.path.join(cache, f"{dataset}__{filename}.feather")
    try: df.to_feather(feather_path)
    except Exception: feather_path = ""
    return {"meta": meta, "head": df.head(5).to_dict(orient="list"), "cache_feather": feather_path}
