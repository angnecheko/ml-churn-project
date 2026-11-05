from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.generate_data import generate_dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "churn.csv"
MODEL_PATH = ROOT_DIR / "models" / "model.pkl"


def train_model(data_path: Path | None = None, model_path: Path | None = None) -> dict:
    resolved_data_path = Path(data_path) if data_path is not None else RAW_DATA_PATH
    resolved_model_path = Path(model_path) if model_path is not None else MODEL_PATH

    if not resolved_data_path.exists():
        generate_dataset()

    df = pd.read_csv(resolved_data_path)
    X = df.drop("churn", axis=1)
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "logistic_regression": Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]
        ),
        "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    results = {}
    for name, model in models.items():
        print(f"\nEntraînement : {name}")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        results[name] = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "f1": f1_score(y_test, predictions),
        }
        print(results[name])

    best_model = models["random_forest"]
    resolved_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, resolved_model_path)
    print(f"\nModèle sauvegardé dans {resolved_model_path}")
    return results


if __name__ == "__main__":
    train_model()
