import os, json, math
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, accuracy_score

LOGS = "/kaggle/working/agent/logs"
REPORTS = "/kaggle/working/agent/reports"
for d in [LOGS, REPORTS]: os.makedirs(d, exist_ok=True)

def _load(df_path: str) -> pd.DataFrame:
    return pd.read_feather(df_path)

def _split_xy(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target]).select_dtypes(include=[np.number]).values.astype(np.float32)
    y = df[target].values
    return X, y

def tabular_baseline(cache_feather: str, cv_path: str, task: str = "auto") -> dict:
    df = _load(cache_feather)
    with open(cv_path, "r") as f: cv = json.load(f)
    target = cv["target"]
    X_all, y_all = _split_xy(df, target)
    if task == "auto":
        if pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > max(20, int(len(df)*0.05)):
            task = "regression"
        else:
            task = "classification"
    scaler = StandardScaler(); X_all = scaler.fit_transform(X_all)
    oof = np.zeros(len(df), dtype=float); folds = []
    for f in cv["folds"]:
        tr, va = np.array(f["train_idx"]), np.array(f["valid_idx"])
        Xtr, Xva = X_all[tr], X_all[va]
        ytr, yva = y_all[tr], y_all[va]
        if task == "regression":
            m = Ridge(alpha=1.0, random_state=42).fit(Xtr, ytr)
            pred = m.predict(Xva); oof[va] = pred
            folds.append({"fold": f["fold"], "rmse": float(math.sqrt(mean_squared_error(yva, pred)))})
        else:
            if not pd.api.types.is_integer_dtype(ytr):
                classes, ytr_enc = np.unique(ytr, return_inverse=True)
                yva_enc = np.searchsorted(classes, yva)
            else:
                classes, ytr_enc = np.unique(ytr, return_inverse=True); yva_enc = yva
            m = LogisticRegression(max_iter=1000, random_state=42).fit(Xtr, ytr_enc)
            pred = m.predict(Xva); oof[va] = pred
            folds.append({"fold": f["fold"], "accuracy": float(accuracy_score(yva_enc, pred))})
    metrics_path = os.path.join(LOGS, "baseline_metrics.json")
    with open(metrics_path, "w") as f: json.dump({"task": task, "folds": folds, "n": int(len(df))}, f, indent=2)
    oof_path = os.path.join(REPORTS, f"oof_{task}.csv"); pd.DataFrame({f"oof_{task}": oof}).to_csv(oof_path, index=False)
    return {"task": task, "fold_metrics": folds, "metrics_path": metrics_path, "oof_path": oof_path}
