import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent

CLASS_NAMES = {
    0: "No_Anemia",
    1: "Mild_HGB_Anemia",
    2: "Iron_Anemia",
    3: "Folate_Anemia",
    4: "B12_Anemia",
}

LEAKAGE_COLS = [
    "All_Class",
    "HGB_Anemia_Class",
    "Iron_anemia_Class",
    "Folate_anemia_class",
    "B12_Anemia_class",
]


def resolve_from_repo(p: Path) -> Path:
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def load_dataframe(xlsx_path: Path) -> pd.DataFrame:
    return pd.read_excel(xlsx_path, engine="openpyxl")


def build_xy(df: pd.DataFrame):
    if "All_Class" not in df.columns:
        raise ValueError("Missing target column: All_Class")

    y = pd.to_numeric(df["All_Class"], errors="coerce").astype("Int64")
    X = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask].astype(int)

    return X, y


def train_model(X: pd.DataFrame, y: pd.Series, test_ratio: float, seed: int):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_ratio,
        random_state=seed,
        stratify=y,
    )

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=seed)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=seed,
                    class_weight={0: 1.0, 1: 25.0, 2: 5.0, 3: 5.0, 4: 5.0},
                    n_jobs=-1,
                ),
            ),
        ]
    )

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    metrics = {
        "testRatio": float(test_ratio),
        "testCount": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macroF1": float(f1_score(y_test, y_pred, average="macro")),
        "confusionMatrix": confusion_matrix(y_test, y_pred).tolist(),
        "trainClassCounts": {int(k): int(v) for k, v in zip(*np.unique(y_train.to_numpy(), return_counts=True))},
    }

    return pipe, metrics


def main():
    p = argparse.ArgumentParser(description="Train anemia classifier (Python)")
    p.add_argument("--xlsx", default="dataset/SKILICARSLAN_Anemia_DataSet.xlsx")
    p.add_argument("--out-model", default="model/anemia_model.joblib")
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    xlsx_path = resolve_from_repo(Path(args.xlsx))
    out_model = resolve_from_repo(Path(args.out_model))

    if not xlsx_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {xlsx_path}\n"
            f"Tip: run from repo root, or pass --xlsx with the full path.\n"
            f"Repo root: {REPO_ROOT}"
        )

    df = load_dataframe(xlsx_path)
    X, y = build_xy(df)

    pipeline, metrics = train_model(X, y, test_ratio=args.test_ratio, seed=args.seed)

    classes = sorted(int(c) for c in np.unique(y.to_numpy()))
    model_bundle = {
        "type": "sklearn_pipeline_logreg",
        "title": "Anemia Classifier",
        "description": "Predicts anemia category from CBC and nutrition markers.",
        "pipeline": pipeline,
        "features": list(X.columns),
        "classes": [{"value": c, "name": CLASS_NAMES.get(c, f"Class_{c}")} for c in classes],
        "negativeClassValue": 0,
        "metrics": metrics,
    }

    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, out_model)

    print(f"Saved model: {out_model}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Test macroF1 : {metrics['macroF1']:.4f}")


if __name__ == "__main__":
    main()
