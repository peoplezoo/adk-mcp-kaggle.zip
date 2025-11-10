import numpy as np, json, os, pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold

CACHE = "/kaggle/working/agent/cache"

def _load_cached_feather(path: str) -> pd.DataFrame:
    if path and os.path.exists(path):
        try: return pd.read_feather(path)
        except Exception: pass
    raise FileNotFoundError("cache not available; rerun dataset_load_csv")

def cv_split(cache_feather: str, target_col: str, n_splits: int = 5, stratified: bool = True) -> dict:
    df = _load_cached_feather(cache_feather)
    if target_col not in df.columns: return {"isError": True, "message": "target missing"}
    y = df[target_col].values
    idxs = np.arange(len(df))
    folds = []
    if stratified and df[target_col].nunique() <= 50 and df[target_col].dtype != float:
        it = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(idxs, y)
    else:
        it = KFold(n_splits=n_splits, shuffle=True, random_state=42).split(idxs)
    for k,(tr,va) in enumerate(it):
        folds.append({"fold": k, "train_idx": tr.tolist(), "valid_idx": va.tolist()})
    path = os.path.join(CACHE, f"cv_{abs(hash((cache_feather,target_col,n_splits,stratified)))}.json")
    with open(path, "w") as f: json.dump({"target": target_col, "folds": folds}, f)
    return {"cv_path": path, "summary": [{"fold": f["fold"], "n_train": len(f["train_idx"]), "n_valid": len(f["valid_idx"])} for f in folds]}
