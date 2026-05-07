import sys

import pandas as pd
import pytest
import yaml

from src.evaluation import hard_datapoints


def _write_config(tmp_path, pred_path):
    ranked_path = tmp_path / "out" / "ranked.csv"
    review_path = tmp_path / "out" / "review_top_k.csv"
    cfg = {
        "input": {"predictions_path": str(pred_path)},
        "weights": {"uncertainty": 0.4, "loss_proxy": 0.3, "cleanlab": 0.3},
        "selection": {"top_k_percent": 0.5},
        "output": {
            "ranked_path": str(ranked_path),
            "review_top_k_path": str(review_path),
        },
    }
    config_path = tmp_path / "hard_points.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return config_path, ranked_path, review_path


def test_hard_datapoints_outputs_expected_columns(tmp_path, monkeypatch):
    df = pd.DataFrame(
        [
            {"id": 1, "true_label": "safe", "pred_label": "safe", "prob_safe": 0.9, "prob_sensitive": 0.1},
            {"id": 2, "true_label": "sensitive", "pred_label": "safe", "prob_safe": 0.6, "prob_sensitive": 0.4},
            {"id": 3, "true_label": "safe", "pred_label": "sensitive", "prob_safe": 0.45, "prob_sensitive": 0.55},
            {"id": 4, "true_label": "sensitive", "pred_label": "sensitive", "prob_safe": 0.05, "prob_sensitive": 0.95},
        ]
    )
    pred_path = tmp_path / "predictions.csv"
    df.to_csv(pred_path, index=False)

    config_path, ranked_path, review_path = _write_config(tmp_path, pred_path)

    monkeypatch.setattr(sys, "argv", ["hard_datapoints", "--config", str(config_path)])
    hard_datapoints.main()

    assert ranked_path.exists()
    assert review_path.exists()

    ranked = pd.read_csv(ranked_path)
    expected_cols = {
        "id",
        "true_label",
        "pred_label",
        "max_prob",
        "uncertainty",
        "loss_proxy",
        "cleanlab_score",
        "hardness",
        "rank",
    }
    assert expected_cols.issubset(set(ranked.columns))
    assert ranked["rank"].tolist() == [1, 2, 3, 4]

    review = pd.read_csv(review_path)
    assert len(review) == 2


def test_hard_datapoints_raises_without_prob_columns(tmp_path, monkeypatch):
    df = pd.DataFrame([
        {"id": 1, "true_label": "safe", "pred_label": "safe"},
        {"id": 2, "true_label": "sensitive", "pred_label": "safe"},
    ])
    pred_path = tmp_path / "predictions.csv"
    df.to_csv(pred_path, index=False)

    config_path, _, _ = _write_config(tmp_path, pred_path)

    monkeypatch.setattr(sys, "argv", ["hard_datapoints", "--config", str(config_path)])
    with pytest.raises(ValueError, match="No probability columns found"):
        hard_datapoints.main()
