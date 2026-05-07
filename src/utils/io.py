import json
from pathlib import Path

import pandas as pd
import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: str | Path, data: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_labeled_data(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() == ".jsonl":
        return pd.read_json(p, lines=True)
    raise ValueError(f"Unsupported data format: {p.suffix}. Use CSV or JSONL.")
