import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent

TARGET_COL = "Diabetes_012"
CLASS_NAMES = {
    0: "No_Diabetes",
    1: "Prediabetes",
    2: "Diabetes",
}


def resolve_from_repo(p: Path) -> Path:
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def build_xy(df: pd.DataFrame):
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    y = pd.to_numeric(df[TARGET_COL], errors="coerce").astype("Int64")
    X = df.drop(columns=[TARGET_COL]).copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask].astype(int)
    return X, y


def stratified_sample(X: pd.DataFrame, y: pd.Series, max_rows: int | None, seed: int):
    if max_rows is None or len(X) <= max_rows:
        return X, y

    _, X_sampled, _, y_sampled = train_test_split(
        X,
        y,
        train_size=max_rows,
        random_state=seed,
        stratify=y,
    )
    return X_sampled.reset_index(drop=True), y_sampled.reset_index(drop=True)


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
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=120,
                    random_state=seed,
                    class_weight={0: 1.0, 1: 4.0, 2: 2.0},
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
        "trainClassCounts": {
            int(k): int(v)
            for k, v in zip(*np.unique(y_train.to_numpy(), return_counts=True))
        },
    }
    return pipe, metrics


def main():
    parser = argparse.ArgumentParser(description="Train diabetes classifier")
    parser.add_argument(
        "--csv",
        default="dataset/diabetes_012_health_indicators_BRFSS2021.csv",
    )
    parser.add_argument("--out-model", default="model/diabetes_model.joblib")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=60000)
    args = parser.parse_args()

    csv_path = resolve_from_repo(Path(args.csv))
    out_model = resolve_from_repo(Path(args.out_model))

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = load_dataframe(csv_path)
    X, y = build_xy(df)
    X, y = stratified_sample(X, y, max_rows=args.max_rows, seed=args.seed)
    pipeline, metrics = train_model(X, y, test_ratio=args.test_ratio, seed=args.seed)

    classes = sorted(int(c) for c in np.unique(y.to_numpy()))
    model_bundle = {
        "type": "sklearn_pipeline_random_forest",
        "title": "Diabetes Risk Classifier",
        "description": "Predicts diabetes status: no diabetes, prediabetes, or diabetes.",
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
