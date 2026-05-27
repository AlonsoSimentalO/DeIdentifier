import pandas as pd
import os

human_dir = "data/human_labelled"
synthetic_path = "data/synthetic/synthetic_labeled.csv"

dfs = []

# Load human-labelled files
for file in os.listdir(human_dir):
    if file.endswith(".csv"):
        path = os.path.join(human_dir, file)
        dfs.append(pd.read_csv(path))

# Load synthetic data
dfs.append(pd.read_csv(synthetic_path))

# Combine
combined = pd.concat(dfs, ignore_index=True)

# Clean
combined = combined.dropna(subset=["text"])

# 🔥 SHUFFLE (IMPORTANT)
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Add IDs AFTER shuffling
combined["id"] = range(len(combined))

# Reorder columns
combined = combined[["id", "text", "label"]]

# Save final dataset
combined.to_csv("data/processed/combined_dataset.csv", index=False)