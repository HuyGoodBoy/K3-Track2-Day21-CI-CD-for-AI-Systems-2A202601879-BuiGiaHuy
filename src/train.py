import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score

EVAL_THRESHOLD = 0.70


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = df.columns

    def _col(*candidates):
        for c in candidates:
            if c in cols:
                return c
        return None

    sulphates = _col("sulphates")
    chlorides = _col("chlorides")
    if sulphates and chlorides:
        df["sulphate_to_chloride"] = df[sulphates] / (df[chlorides] + 1e-9)

    free_so2 = _col("free sulfur dioxide", "free_sulfur_dioxide")
    total_so2 = _col("total sulfur dioxide", "total_sulfur_dioxide")
    if free_so2 and total_so2:
        df["free_to_total_so2"] = df[free_so2] / (df[total_so2] + 1e-9)

    density = _col("density")
    alcohol = _col("alcohol")
    if density and alcohol:
        df["density_to_alcohol"] = df[density] / (df[alcohol] + 1e-9)

    acid_cols = [
        c for c in (
            _col("fixed acidity", "fixed_acidity"),
            _col("volatile acidity", "volatile_acidity"),
            _col("citric acid", "citric_acid"),
        ) if c is not None
    ]
    if len(acid_cols) >= 2:
        df["acidity_sum"] = df[acid_cols].sum(axis=1)

    sugar = _col("residual sugar", "residual_sugar")
    if sugar and alcohol:
        df["sugar_to_alcohol"] = df[sugar] / (df[alcohol] + 1e-9)

    ph = _col("pH")
    if ph and "acidity_sum" in df.columns:
        df["ph_times_acidity"] = df[ph] * df["acidity_sum"]

    if free_so2 and total_so2 and alcohol:
        df["so2_per_alcohol"] = (df[total_so2] + df[free_so2]) / (df[alcohol] + 1e-9)

    return df


def train(
    params: dict,
    data_path: str = "data/train_phase2.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.
    """

    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    df_train = _add_features(df_train)
    df_eval = _add_features(df_eval)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    with mlflow.start_run():

        mlflow.log_params(params)

        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        hgb = HistGradientBoostingClassifier(
            max_iter=500,
            max_depth=8,
            learning_rate=0.05,
            random_state=42,
        )

        model = VotingClassifier(
            estimators=[("rf", rf), ("hgb", hgb)],
            voting="soft",
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/metrics.json", "w") as f:
            json.dump({"accuracy": acc, "f1_score": f1}, f)

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)