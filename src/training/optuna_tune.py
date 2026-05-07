import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import optuna
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from src.utils.io import load_labeled_data, load_yaml, save_json
from src.utils.paths import ensure_dir, ensure_parent_dir
from src.utils.seeding import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for HF training")
    parser.add_argument("--config", type=str, default="configs/optuna.yaml")
    return parser.parse_args()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
    }


def prepare_data(cfg: dict):
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    seed = int(cfg["seed"])

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
    max_length = int(model_cfg["max_length"])

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))

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

    return train_ds, val_ds, tokenizer, labels, label2id, id2label


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))

    mlflow_cfg = cfg["mlflow"]
    search_cfg = cfg["search"]
    space = cfg["space"]
    fixed = cfg["fixed_training"]
    out_cfg = cfg["output"]
    model_cfg = cfg["model"]

    ensure_parent_dir(out_cfg["best_params_path"])
    ensure_parent_dir(out_cfg["trials_path"])
    ensure_dir(out_cfg["temp_model_dir"])

    train_ds, val_ds, tokenizer, labels, label2id, id2label = prepare_data(cfg)

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    trial_rows = []

    def objective(trial: optuna.Trial) -> float:
        learning_rate = trial.suggest_float(
            "learning_rate",
            float(space["learning_rate"]["low"]),
            float(space["learning_rate"]["high"]),
            log=bool(space["learning_rate"].get("log", True)),
        )
        batch_size = trial.suggest_categorical("batch_size", list(space["batch_size"]["choices"]))
        epochs = trial.suggest_int("epochs", int(space["epochs"]["low"]), int(space["epochs"]["high"]))
        weight_decay = trial.suggest_float("weight_decay", float(space["weight_decay"]["low"]), float(space["weight_decay"]["high"]))

        model = AutoModelForSequenceClassification.from_pretrained(
            model_cfg["model_name"],
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
        )

        run_name = f"trial_{trial.number}"
        with mlflow.start_run(run_name=run_name, nested=False):
            mlflow.log_param("trial_number", trial.number)
            mlflow.log_params(
                {
                    "model_name": model_cfg["model_name"],
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "weight_decay": weight_decay,
                    "max_length": int(model_cfg["max_length"]),
                }
            )

            training_args = TrainingArguments(
                output_dir=str(Path(out_cfg["temp_model_dir"]) / f"trial_{trial.number}"),
                learning_rate=learning_rate,
                per_device_train_batch_size=int(batch_size),
                per_device_eval_batch_size=int(batch_size),
                num_train_epochs=float(epochs),
                weight_decay=weight_decay,
                eval_strategy=fixed["eval_strategy"],
                save_strategy=fixed["save_strategy"],
                logging_steps=int(fixed["logging_steps"]),
                report_to=[],
                seed=int(cfg["seed"]),
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                processing_class=tokenizer,
                compute_metrics=compute_metrics,
            )

            trainer.train()
            eval_metrics = trainer.evaluate()

            metric_key = search_cfg["metric_to_optimize"]
            objective_value = float(eval_metrics.get(metric_key, float("nan")))

            mlflow.log_metrics({
                "eval_loss": float(eval_metrics.get("eval_loss", float("nan"))),
                "eval_accuracy": float(eval_metrics.get("eval_accuracy", float("nan"))),
                "eval_f1_macro": float(eval_metrics.get("eval_f1_macro", float("nan"))),
            })

            trial_rows.append(
                {
                    "trial_number": trial.number,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "weight_decay": weight_decay,
                    "eval_loss": float(eval_metrics.get("eval_loss", float("nan"))),
                    "eval_accuracy": float(eval_metrics.get("eval_accuracy", float("nan"))),
                    "eval_f1_macro": float(eval_metrics.get("eval_f1_macro", float("nan"))),
                    "objective": objective_value,
                }
            )

            return objective_value

    study = optuna.create_study(direction=search_cfg["direction"])
    study.optimize(objective, n_trials=int(search_cfg["n_trials"]))

    best = {
        "best_trial": int(study.best_trial.number),
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "metric": search_cfg["metric_to_optimize"],
        "direction": search_cfg["direction"],
    }

    save_json(out_cfg["best_params_path"], best)
    pd.DataFrame(trial_rows).to_csv(out_cfg["trials_path"], index=False)

    print(json.dumps({"status": "ok", "best": best}, indent=2))


if __name__ == "__main__":
    main()
