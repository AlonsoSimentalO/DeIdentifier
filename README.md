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
