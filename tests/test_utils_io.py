import json

import pandas as pd
import pytest

from src.utils.io import load_labeled_data, save_json


def test_save_json_writes_file(tmp_path):
    out = tmp_path / "nested" / "metrics.json"
    payload = {"acc": 0.9, "f1": 0.8}

    save_json(out, payload)

    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == payload


def test_load_labeled_data_csv_and_jsonl(tmp_path):
    df = pd.DataFrame(
        [
            {"id": 1, "text": "a", "label": "x"},
            {"id": 2, "text": "b", "label": "y"},
        ]
    )

    csv_path = tmp_path / "sample.csv"
    jsonl_path = tmp_path / "sample.jsonl"
    txt_path = tmp_path / "sample.txt"

    df.to_csv(csv_path, index=False)
    df.to_json(jsonl_path, orient="records", lines=True)
    txt_path.write_text("not supported", encoding="utf-8")

    loaded_csv = load_labeled_data(csv_path)
    loaded_jsonl = load_labeled_data(jsonl_path)

    assert list(loaded_csv.columns) == ["id", "text", "label"]
    assert len(loaded_csv) == 2
    assert len(loaded_jsonl) == 2

    with pytest.raises(ValueError, match="Unsupported data format"):
        load_labeled_data(txt_path)
