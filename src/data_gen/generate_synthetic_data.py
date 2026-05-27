import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from faker import Faker


@dataclass
class GenerationConfig:
    output_path: str
    n_samples: int
    sensitive_ratio: float
    locale: str
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic labeled documents (text,label)")
    parser.add_argument("--output", type=str, default="data/synthetic/synthetic_labeled.csv")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--sensitive-ratio", type=float, default=0.5)
    parser.add_argument("--locale", type=str, default="en_US")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _sensitive_templates() -> list[str]:
    return [
        "Patient record: {name}, email {email}, phone {phone}. Follow-up date: {date}.",
        "Internal HR note for employee {name}. National ID: {id_number}. Address: {address}.",
        "Customer complaint from {name}. Card ending: {card_last4}. Full card: {card}.",
        "Travel request by {name}. Passport number: {passport}. Emergency contact: {phone}.",
        "Loan application: {name}, account {iban}, annual salary CHF {salary}.",
        "Support ticket. User {username} forgot password. Backup email: {email}.",
        "Invoice to {name}. Billing address: {address}. Tax ID: {id_number}.",
        "Insurance claim from {name}. Policy ID: {policy}. Contact {phone}.",
    ]


def _non_sensitive_templates() -> list[str]:
    return [
        "Project update: sprint goals were completed and backlog items reprioritized.",
        "Team meeting summary: discussed milestones, blockers, and release timeline.",
        "Weather report: moderate rain expected this afternoon with cooler temperatures.",
        "Technical note: API latency improved after database index optimization.",
        "Education notice: assignment deadline moved to next Friday.",
        "Operations memo: warehouse inventory cycle count scheduled for Monday.",
        "General announcement: office kitchen maintenance planned this weekend.",
        "Status report: deployment finished successfully with no critical incidents.",
    ]


def _fake_values(fake: Faker) -> dict:
    return {
        "name": fake.name(),
        "email": fake.email(),
        "phone": fake.phone_number(),
        "date": fake.date(),
        "id_number": fake.bothify(text="??########"),
        "address": fake.address().replace("\\n", ", "),
        "card": fake.credit_card_number(),
        "card_last4": fake.credit_card_number()[-4:],
        "passport": fake.bothify(text="??######"),
        "iban": fake.iban(),
        "salary": fake.random_int(min=50000, max=180000),
        "username": fake.user_name(),
        "policy": fake.bothify(text="POL-######"),
    }


def generate_dataset(cfg: GenerationConfig) -> pd.DataFrame:
    fake = Faker(cfg.locale)
    Faker.seed(cfg.seed)
    random.seed(cfg.seed)

    sens_templates = _sensitive_templates()
    nons_templates = _non_sensitive_templates()

    n_sensitive = int(round(cfg.n_samples * cfg.sensitive_ratio))
    n_non_sensitive = cfg.n_samples - n_sensitive

    rows = []

    for _ in range(n_sensitive):
        template = random.choice(sens_templates)
        text = template.format(**_fake_values(fake))
        rows.append({"text": text, "label": "sensitive"})

    for _ in range(n_non_sensitive):
        template = random.choice(nons_templates)
        text = template
        rows.append({"text": text, "label": "non_sensitive"})

    random.shuffle(rows)

    return pd.DataFrame(rows, columns=["text", "label"])


def main() -> None:
    args = parse_args()

    if not (0.0 < args.sensitive_ratio < 1.0):
        raise ValueError("--sensitive-ratio must be between 0 and 1 (exclusive)")
    if args.n_samples < 20:
        raise ValueError("--n-samples must be >= 20 for meaningful training split")

    cfg = GenerationConfig(
        output_path=args.output,
        n_samples=args.n_samples,
        sensitive_ratio=args.sensitive_ratio,
        locale=args.locale,
        seed=args.seed,
    )

    df = generate_dataset(cfg)

    out = Path(cfg.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    label_counts = df["label"].value_counts().to_dict()
    print(f"Saved synthetic dataset: {out}")
    print(f"Rows: {len(df)}")
    print(f"Label distribution: {label_counts}")


if __name__ == "__main__":
    main()
