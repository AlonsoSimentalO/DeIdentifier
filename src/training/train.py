import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.utils.io import load_labeled_data, load_yaml, save_json
from src.utils.paths import ensure_dir, ensure_parent_dir
from src.utils.seeding import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune text classifier")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    return parser.parse_args()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
    }


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    seed = int(cfg["seed"])
    set_seed(seed)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    out_cfg = cfg["output"]
    mlflow_cfg = cfg["mlflow"]

    df = load_labeled_data(data_cfg["input_path"])
    id_col = data_cfg["id_column"]
    text_col = data_cfg["text_column"]
    label_col = data_cfg["label_column"]

    required_cols = {id_col, text_col, label_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    labels = sorted(df[label_col].astype(str).unique().tolist())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    df = df[[id_col, text_col, label_col]].copy()
    df[label_col] = df[label_col].astype(str)
    df["label_id"] = df[label_col].map(label2id)

    train_df, val_df = train_test_split(
        df,
        test_size=float(data_cfg["test_size"]),
        random_state=seed,
        stratify=df["label_id"] if len(labels) > 1 and len(df) >= len(labels) * 2 else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_name"])

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))

    max_length = int(model_cfg["max_length"])

    def tokenize_batch(batch):
        return tokenizer(batch[text_col], truncation=True, padding="max_length", max_length=max_length)

    train_ds = train_ds.map(tokenize_batch, batched=True)
    val_ds = val_ds.map(tokenize_batch, batched=True)

    train_ds = train_ds.rename_column("label_id", "labels")
    val_ds = val_ds.rename_column("label_id", "labels")

    keep_cols = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in train_ds.column_names:
        keep_cols.append("token_type_ids")
    train_ds.set_format(type="torch", columns=keep_cols)
    val_ds.set_format(type="torch", columns=keep_cols)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["model_name"],
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    ensure_dir(out_cfg["base_dir"])
    ensure_dir(out_cfg["model_dir"])
    ensure_parent_dir(out_cfg["predictions_path"])
    ensure_parent_dir(out_cfg["metrics_path"])

    training_args = TrainingArguments(
        output_dir=out_cfg["model_dir"],
        learning_rate=float(train_cfg["learning_rate"]),
        per_device_train_batch_size=int(train_cfg["batch_size"]),
        per_device_eval_batch_size=int(train_cfg["batch_size"]),
        num_train_epochs=float(train_cfg["epochs"]),
        weight_decay=float(train_cfg["weight_decay"]),
        eval_strategy=train_cfg["eval_strategy"],
        save_strategy=train_cfg["save_strategy"],
        logging_steps=int(train_cfg["logging_steps"]),
        report_to=[],
        seed=seed,
    )

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    with mlflow.start_run():
        mlflow.log_params(
            {
                "model_name": model_cfg["model_name"],
                "max_length": max_length,
                "learning_rate": float(train_cfg["learning_rate"]),
                "batch_size": int(train_cfg["batch_size"]),
                "epochs": float(train_cfg["epochs"]),
                "weight_decay": float(train_cfg["weight_decay"]),
                "seed": seed,
            }
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
        )

        train_result = trainer.train()
        eval_metrics = trainer.evaluate()

        pred_output = trainer.predict(val_ds)
        logits = pred_output.predictions
        probs = softmax(logits)
        pred_ids = np.argmax(logits, axis=1)

        val_export = val_df.reset_index(drop=True).copy()
        val_export["true_label"] = val_export[label_col]
        val_export["pred_label"] = [id2label[int(i)] for i in pred_ids]
        val_export["max_prob"] = probs.max(axis=1)

        for class_idx, class_name in id2label.items():
            val_export[f"prob_{class_name}"] = probs[:, int(class_idx)]

        pred_path = Path(out_cfg["predictions_path"])
        val_export.to_csv(pred_path, index=False)

        metrics = {
            "train_loss": float(train_result.training_loss),
            "eval_loss": float(eval_metrics.get("eval_loss", float("nan"))),
            "eval_accuracy": float(eval_metrics.get("eval_accuracy", float("nan"))),
            "eval_f1_macro": float(eval_metrics.get("eval_f1_macro", float("nan"))),
        }

        save_json(out_cfg["metrics_path"], metrics)

        model.save_pretrained(out_cfg["model_dir"])
        tokenizer.save_pretrained(out_cfg["model_dir"])

        mlflow.log_metrics(metrics)
        mlflow.log_artifact(args.config)
        mlflow.log_artifact(str(pred_path))
        mlflow.log_artifact(out_cfg["metrics_path"])

        label_map_path = Path(out_cfg["base_dir"]) / "reports" / "label_mapping.json"
        save_json(label_map_path, {"label2id": label2id, "id2label": id2label})
        mlflow.log_artifact(str(label_map_path))

        print(json.dumps({"status": "ok", "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
