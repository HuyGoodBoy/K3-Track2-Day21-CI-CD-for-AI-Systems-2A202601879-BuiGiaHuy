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
    df["sulphate_to_chloride"] = df["sulphates"] / (df["chlorides"] + 1e-9)
    df["free_to_total_so2"] = df["free sulfur dioxide"] / (df["total sulfur dioxide"] + 1e-9)
    df["density_to_alcohol"] = df["density"] / (df["alcohol"] + 1e-9)
    df["acidity_sum"] = df["fixed acidity"] + df["volatile acidity"] + df["citric acid"]
    df["sugar_to_alcohol"] = df["residual sugar"] / (df["alcohol"] + 1e-9)
    df["ph_times_acidity"] = df["pH"] * df["acidity_sum"]
    df["so2_per_alcohol"] = (df["total sulfur dioxide"] + df["free sulfur dioxide"]) / (df["alcohol"] + 1e-9)
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