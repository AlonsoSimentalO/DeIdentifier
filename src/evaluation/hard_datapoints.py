import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from cleanlab.rank import get_label_quality_scores

from src.utils.io import load_yaml
from src.utils.paths import ensure_parent_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank hard datapoints")
    parser.add_argument("--config", type=str, default="configs/hard_points.yaml")
    parser.add_argument("--predictions", type=str, default=None)
    return parser.parse_args()


def _safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return -np.log(np.clip(x, eps, 1.0))


def _minmax(arr: np.ndarray) -> np.ndarray:
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if np.isclose(lo, hi):
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    pred_path = args.predictions or cfg["input"]["predictions_path"]
    df = pd.read_csv(pred_path)

    required = {"id", "true_label", "pred_label"}
    missing_required = required - set(df.columns)
    if missing_required:
        raise ValueError(f"Missing required columns in predictions file: {sorted(missing_required)}")

    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if not prob_cols:
        raise ValueError("No probability columns found. Expected columns like prob_<class>.")

    probs = df[prob_cols].to_numpy(dtype=float)
    probs = probs / probs.sum(axis=1, keepdims=True)

    classes = [c.replace("prob_", "", 1) for c in prob_cols]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    if not set(df["true_label"].astype(str).unique()).issubset(class_to_idx.keys()):
        unknown = sorted(set(df["true_label"].astype(str).unique()) - set(class_to_idx.keys()))
        raise ValueError(f"Unknown labels in true_label not present in prob_ columns: {unknown}")

    true_idx = df["true_label"].astype(str).map(class_to_idx).to_numpy(dtype=int)

    max_prob = probs.max(axis=1)
    uncertainty = 1.0 - max_prob
    p_true = probs[np.arange(len(probs)), true_idx]
    loss_proxy = _safe_log(p_true)

    quality_scores = get_label_quality_scores(labels=true_idx, pred_probs=probs)
    cleanlab_score = 1.0 - quality_scores

    w = cfg["weights"]
    hardness = (
        float(w["uncertainty"]) * _minmax(uncertainty)
        + float(w["loss_proxy"]) * _minmax(loss_proxy)
        + float(w["cleanlab"]) * _minmax(cleanlab_score)
    )

    out = df.copy()
    out["max_prob"] = max_prob
    out["uncertainty"] = uncertainty
    out["loss_proxy"] = loss_proxy
    out["cleanlab_score"] = cleanlab_score
    out["hardness"] = hardness

    out = out.sort_values("hardness", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)

    top_k_percent = float(cfg["selection"]["top_k_percent"])
    k = max(1, int(np.ceil(len(out) * top_k_percent)))
    review = out.head(k).copy()

    ranked_path = Path(cfg["output"]["ranked_path"])
    review_path = Path(cfg["output"]["review_top_k_path"])
    ensure_parent_dir(ranked_path)
    ensure_parent_dir(review_path)

    out.to_csv(ranked_path, index=False)
    review.to_csv(review_path, index=False)

    print(f"Saved ranked hard datapoints: {ranked_path}")
    print(f"Saved top-k review subset ({k} rows): {review_path}")


if __name__ == "__main__":
    main()
