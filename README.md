# mlops-privacy-pipeline

Initial project repository. Base structure only, no implementation yet.

## Pitch Deck (summary)
- **Challenge:** GDPR rules are getting stricter, while the use of cloud-based AI tools keeps growing.
- **Solution:** Build a streamlined process to train and set up an **in-house** tool that removes sensitive data from documents.
- **Approach:** Synthetic data + human-in-the-loop labeling.
- **Pipeline:** Synthetic data generation -> human validation -> quality filtering -> fine-tuning (DistilBERT/RoBERTa) -> hard datapoint detection -> evaluation and retraining.
- **Mentioned tooling:** Label Studio, HuggingFace, MLflow, DVC, Docker.

## Team Responsibilities
- **Alexander:** Front page / Goal & Data, Tools & Pipeline overview, Labeling.
- **Joel:** Interface / UI.
- **Oscar:** Model training (HuggingFace) and hard datapoint detection.
- **Curtis:** Data versioning (DVC).

## Oscar module commands
- Create/activate venv:
  - PowerShell: `python -m venv .venv; .\.venv\Scripts\Activate.ps1`
  - Git Bash: `python -m venv .venv && source .venv/Scripts/activate`
- Install dependencies: `python -m pip install -r requirements.txt`
- Train baseline: `python -m src.training.train --config configs/train.yaml`
- Optuna tuning: `python -m src.training.optuna_tune --config configs/optuna.yaml`
- Rank hard datapoints: `python -m src.evaluation.hard_datapoints --config configs/hard_points.yaml`
