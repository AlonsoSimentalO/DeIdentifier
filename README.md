# mlops-privacy-pipeline

This repository contains the project pipeline for sensitive-data detection using synthetic data, model fine-tuning, and hard-sample review.

## Project Goal
- Build an in-house workflow to detect sensitive information in text documents.
- Use synthetic data to bootstrap training.
- Keep a human-in-the-loop improvement cycle through hard datapoint review.

## Current Scope Implemented
- Synthetic dataset generation (`id,text,label`) with Faker-based templates.
- Text classification fine-tuning with HuggingFace DistilBERT.
- Hyperparameter search with Optuna.
- Experiment tracking with MLflow.
- Hard datapoint ranking using confidence, loss proxy, and Cleanlab scores.

## Commands To Run The Project
- Create/activate virtual environment:
  - PowerShell: `python -m venv .venv; .\.venv\Scripts\Activate.ps1`
  - Git Bash: `python -m venv .venv && source .venv/Scripts/activate`
- Install dependencies: `python -m pip install -r requirements.txt`
- old:
  - Generate synthetic data: `python -m src.data_gen.generate_synthetic_data --output data/processed/synthetic_labeled.csv --n-samples 600 --sensitive-ratio 0.5`
  - Train model: `python -m src.training.train --config configs/train.yaml`
  - Run hyperparameter tuning: `python -m src.training.optuna_tune --config configs/optuna.yaml`
  - Rank hard datapoints: `python -m src.evaluation.hard_datapoints --config configs/hard_points.yaml`
- new:
  - streamlit run Deidentifier_app.py

## Key Outputs
- Trained model/checkpoint: `artifacts/model/`
- Training metrics: `artifacts/reports/train_metrics.json`
- Optuna trial results: `artifacts/reports/optuna_trials.csv`
- Best Optuna parameters: `artifacts/reports/optuna_best_params.json`
- Ranked hard samples: `artifacts/hard_points/hard_datapoints_ranked.csv`
- Top-k review subset: `artifacts/hard_points/review_top_k.csv`
