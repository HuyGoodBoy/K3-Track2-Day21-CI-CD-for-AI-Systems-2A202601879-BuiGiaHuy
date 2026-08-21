import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

# Bonus 1: DagsHub MLflow remote tracking
# If MLFLOW_TRACKING_URI is set (CI/CD or local), use it. Otherwise use local sqlite.
# Also requires DAGSHUB_USER + DAGSHUB_TOKEN. If either is empty, we fall back to local sqlite
# so the pipeline doesn't break if the user hasn't configured DagsHub secrets yet.
_DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
_dagshub_user = os.environ.get("DAGSHUB_USER", "").strip()
_dagshub_token = os.environ.get("DAGSHUB_TOKEN", "").strip()

if _tracking_uri and _dagshub_user and _dagshub_token:
    mlflow.set_tracking_uri(_tracking_uri)
    os.environ["MLFLOW_TRACKING_USERNAME"] = _dagshub_user
    os.environ["MLFLOW_TRACKING_PASSWORD"] = _dagshub_token
    print(f"[Bonus 1] MLflow tracking URI set to DagsHub: {_tracking_uri}")
    print(f"[Bonus 1] DagsHub user: {_dagshub_user}")
else:
    mlflow.set_tracking_uri(_DEFAULT_TRACKING_URI)
    print(f"[Bonus 1] MLflow tracking URI = local sqlite (DagsHub secrets missing or empty)")
    print(f"           MLFLOW_TRACKING_URI={_tracking_uri!r}, DAGSHUB_USER={_dagshub_user!r}, DAGSHUB_TOKEN set={bool(_dagshub_token)}")

EVAL_THRESHOLD = 0.70
DRIFT_MIN_RATIO = 0.10


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


def _build_model(model_type: str, params: dict):
    """Builds the requested model. Supports 4 types for Bonus 2."""
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", None),
            min_samples_split=params.get("min_samples_split", 2),
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 5),
            learning_rate=params.get("learning_rate", 0.05),
            random_state=42,
        )

    if model_type == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=params.get("max_iter", 1000),
                class_weight="balanced",
                multi_class="multinomial",
                random_state=42,
            )),
        ])

    if model_type == "xgboost":
        return XGBClassifier(
            n_estimators=params.get("n_estimators", 500),
            max_depth=params.get("max_depth", 8),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )

    if model_type == "voting":
        xgb = XGBClassifier(
            n_estimators=params.get("n_estimators", 500),
            max_depth=params.get("max_depth", 8),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )
        rf = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        return VotingClassifier(
            estimators=[("xgb", xgb), ("rf", rf)],
            voting="soft",
            n_jobs=1,
        )

    raise ValueError(f"Unknown model_type: {model_type}")


def _check_label_distribution(y: pd.Series, min_ratio: float = DRIFT_MIN_RATIO) -> dict:
    """Bonus 5: data drift detection. Returns distribution dict."""
    counts = y.value_counts().sort_index()
    total = len(y)
    dist = {int(k): int(v) for k, v in counts.items()}
    ratios = {int(k): float(v) / total for k, v in counts.items()}
    warnings = []
    for cls, r in ratios.items():
        if r < min_ratio:
            warnings.append(
                f"WARNING: class {cls} counts for only {r:.1%} of samples "
                f"(below {min_ratio:.0%} threshold)"
            )
    return {
        "counts": dist,
        "ratios": ratios,
        "min_ratio": min_ratio,
        "warnings": warnings,
    }


def _write_report(metrics: dict, cm: np.ndarray, y_eval, preds, dist_info: dict) -> str:
    """Bonus 3: write a text report with confusion matrix + per-class metrics."""
    lines = []
    lines.append("=" * 60)
    lines.append("MODEL PERFORMANCE REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Accuracy : {metrics['accuracy']:.4f}")
    lines.append(f"F1 (weighted): {metrics['f1_score']:.4f}")
    lines.append("")
    lines.append("-" * 60)
    lines.append("LABEL DISTRIBUTION (Bonus 5)")
    lines.append("-" * 60)
    for cls, ratio in dist_info["ratios"].items():
        lines.append(f"  class {cls}: {dist_info['counts'][cls]:5d} samples ({ratio:.1%})")
    if dist_info["warnings"]:
        lines.append("")
        for w in dist_info["warnings"]:
            lines.append(f"  {w}")
    else:
        lines.append(f"  OK: all classes >= {dist_info['min_ratio']:.0%} threshold")
    lines.append("")
    lines.append("-" * 60)
    lines.append("CONFUSION MATRIX (rows=true, cols=pred)")
    lines.append("-" * 60)
    lines.append("        " + "  ".join(f"pred_{i}" for i in range(cm.shape[0])))
    for i, row in enumerate(cm):
        lines.append(f"true_{i}   " + "  ".join(f"{v:6d}" for v in row))
    lines.append("")
    lines.append("-" * 60)
    lines.append("PER-CLASS PRECISION / RECALL / F1")
    lines.append("-" * 60)
    lines.append(classification_report(y_eval, preds, digits=4))
    lines.append("-" * 60)

    return "\n".join(lines)


def train(
    params: dict,
    data_path: str = "data/train_phase2.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.
    Supports multiple model types (Bonus 2), writes report (Bonus 3),
    detects data drift (Bonus 5).
    """

    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    df_train = _add_features(df_train)
    df_eval = _add_features(df_eval)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # Bonus 5: data drift check on train labels
    dist_info = _check_label_distribution(y_train)
    if dist_info["warnings"]:
        for w in dist_info["warnings"]:
            print(w)
    else:
        print(
            f"[Bonus 5] OK: all classes >= {dist_info['min_ratio']:.0%} "
            f"min ratio (smallest class = {min(dist_info['ratios'].values()):.1%})"
        )

    # Bonus 2: choose model_type from params
    model_type = params.get("model_type", "voting")
    print(f"[Bonus 2] Using model_type = {model_type}")
    model = _build_model(model_type, params)

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param("model_type", model_type)

        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")
        precision = precision_score(y_eval, preds, average="weighted", zero_division=0)
        recall = recall_score(y_eval, preds, average="weighted", zero_division=0)
        cm = confusion_matrix(y_eval, preds)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        for cls, ratio in dist_info["ratios"].items():
            mlflow.log_metric(f"label_ratio_class_{cls}", ratio)

        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        os.makedirs("outputs", exist_ok=True)
        metrics_full = {
            "accuracy": acc,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "model_type": model_type,
            "label_distribution": dist_info["counts"],
            "label_ratios": dist_info["ratios"],
        }
        with open("outputs/metrics.json", "w") as f:
            json.dump(metrics_full, f, indent=2)

        # Bonus 3: write report.txt
        report = _write_report(metrics_full, cm, y_eval, preds, dist_info)
        with open("outputs/report.txt", "w") as f:
            f.write(report)
        print("\n" + report)

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
