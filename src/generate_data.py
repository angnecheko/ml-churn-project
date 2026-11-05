from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "churn.csv"


def generate_dataset() -> pd.DataFrame:
    np.random.seed(42)

    n = 2000

    age = np.random.randint(18, 70, n)
    anciennete = np.random.randint(1, 120, n)
    revenu = np.random.randint(150000, 1000000, n)
    nb_reclamations = np.random.poisson(2, n)
    montant_mensuel = np.random.randint(5000, 75000, n)
    retards_paiement = np.random.poisson(1, n)

    score = (
        0.04 * nb_reclamations
        + 0.08 * retards_paiement
        - 0.015 * anciennete
        + 0.00001 * montant_mensuel
    )

    probabilite = 1 / (1 + np.exp(-score))
    churn = (np.random.random(n) < probabilite).astype(int)

    df = pd.DataFrame({
        "age": age,
        "anciennete": anciennete,
        "revenu": revenu,
        "nb_reclamations": nb_reclamations,
        "montant_mensuel": montant_mensuel,
        "retards_paiement": retards_paiement,
        "churn": churn,
    })

    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    print(df.head())
    print()
    print(df["churn"].value_counts())