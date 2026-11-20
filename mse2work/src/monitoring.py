
import json
import warnings
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from .config import (
    DATA_PROCESSED,
    MODELS_DIR,
    REPORTS_DIR,
    FEATURE_COLUMNS,
    RANDOM_STATE
)

from .data import (
    load_training_frame,
    load_incoming_frame,
    feature_frame
)


def metrics(model, X, y):
    predictions = model.predict(X)

    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y, predictions, zero_division=0)
        )
    }


def build_classifier():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


def main():

    # Create required directories
    for folder in (DATA_PROCESSED, MODELS_DIR, REPORTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    old = load_training_frame()
    incoming = load_incoming_frame()

    # ==================================================
    # TASK A — INCOMING DATA VALIDATION
    # ==================================================

    required = (
        ["engine_id", "cycle"]
        + FEATURE_COLUMNS
        + ["failure_soon"]
    )

    missing_cols = [
        column for column in required
        if column not in incoming.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}"
        )

    duplicate_count = int(
        incoming.duplicated(
            subset=["engine_id", "cycle"]
        ).sum()
    )

    missing_values = (
        incoming[required]
        .isna()
        .sum()
        .to_dict()
    )

    incoming = incoming.drop_duplicates(
        subset=["engine_id", "cycle"]
    )

    incoming = incoming.dropna(
        subset=required
    )

    if incoming["failure_soon"].nunique() < 2:
        raise ValueError(
            "Incoming data must contain both target classes."
        )

    incoming.to_csv(
        DATA_PROCESSED / "incoming_batch.csv",
        index=False
    )

    print("\nTASK A — INCOMING DATA VALIDATION")

    print(json.dumps({
        "rows": int(len(incoming)),
        "engines": int(incoming["engine_id"].nunique()),
        "duplicate_records": duplicate_count,
        "missing_values": {
            key: int(value)
            for key, value in missing_values.items()
        },
        "target_distribution": {
            str(key): int(value)
            for key, value in
            incoming["failure_soon"].value_counts().items()
        }
    }, indent=2))

    # Split incoming data by engine to prevent data leakage
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=RANDOM_STATE
    )

    train_idx, eval_idx = next(
        splitter.split(
            incoming,
            groups=incoming["engine_id"]
        )
    )

    new_train = incoming.iloc[train_idx]
    new_eval = incoming.iloc[eval_idx]

    if (
        new_train["failure_soon"].nunique() < 2
        or new_eval["failure_soon"].nunique() < 2
    ):
        raise ValueError(
            "Both training and evaluation splits must contain "
            "both target classes."
        )

    X_eval = feature_frame(new_eval)
    y_eval = new_eval["failure_soon"].astype(int)

    # ==================================================
    # CHAMPION MODEL — REBUILD BASELINE
    # ==================================================

    old_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=RANDOM_STATE
    )

    old_train_idx, _ = next(
        old_splitter.split(
            old,
            groups=old["engine_id"]
        )
    )

    champion = build_classifier()

    champion.fit(
        feature_frame(old.iloc[old_train_idx]),
        old.iloc[old_train_idx]["failure_soon"].astype(int)
    )

    champion_path = (
        MODELS_DIR / "champion_model_rebuilt.joblib"
    )

    joblib.dump(champion, champion_path)

    # ==================================================
    # TASK B — CHAMPION EVALUATION
    # ==================================================

    champion_metrics = metrics(
        champion,
        X_eval,
        y_eval
    )

    print("\nTASK B — CHAMPION EVALUATION")
    print(json.dumps(champion_metrics, indent=2))

    # ==================================================
    # TASK C — CANDIDATE RETRAINING
    # ==================================================

    combined = pd.concat(
        [old, new_train],
        ignore_index=True
    )

    X_train = feature_frame(combined)
    y_train = combined["failure_soon"].astype(int)

    candidate = build_classifier()

    candidate.fit(
        X_train,
        y_train
    )

    candidate_path = (
        MODELS_DIR / "candidate_model.joblib"
    )

    joblib.dump(candidate, candidate_path)

    candidate_metrics = metrics(
        candidate,
        X_eval,
        y_eval
    )

    print("\nTASK C — CANDIDATE MODEL")
    print("Combined training rows:", len(combined))
    print("Candidate saved:", candidate_path)
    print(json.dumps(candidate_metrics, indent=2))

    # ==================================================
    # TASK C — OPTIONAL MLFLOW TRACKING
    # ==================================================

    mlflow_status = "not_logged"

    try:
        import mlflow
        import mlflow.sklearn

        tracking_db = (
            REPORTS_DIR / "mlflow.db"
        ).as_posix()

        mlflow.set_tracking_uri(
            f"sqlite:///{tracking_db}"
        )

        mlflow.set_experiment(
            "nasa-fd001-model-quality-monitoring"
        )

        with mlflow.start_run(
            run_name="candidate-retraining"
        ):

            mlflow.log_params({
                "dataset": "NASA C-MAPSS FD001",
                "algorithm": "RandomForestClassifier",
                "n_estimators": 300,
                "max_depth": 14,
                "random_state": RANDOM_STATE,
                "incoming_train_engines":
                    int(new_train["engine_id"].nunique()),
                "evaluation_engines":
                    int(new_eval["engine_id"].nunique())
            })

            mlflow.log_metrics({
                f"candidate_{key}": value
                for key, value in candidate_metrics.items()
            })

            mlflow.sklearn.log_model(
                sk_model=candidate,
                name="candidate_model",
                skops_trusted_types=[
                    "sklearn.tree._tree.Tree"
                ]
            )

        mlflow_status = "logged"

        print("\nMLflow tracking: SUCCESS")

    except Exception as exc:

        mlflow_status = (
            f"not_logged: {type(exc).__name__}: {exc}"
        )

        warnings.warn(
            f"MLflow tracking skipped: {mlflow_status}"
        )

    # ==================================================
    # TASK D — QUALITY GATE
    # ==================================================

    f1_pass = (
        candidate_metrics["f1"]
        >= champion_metrics["f1"]
    )

    recall_pass = (
        candidate_metrics["recall"]
        >= champion_metrics["recall"]
    )

    eligible = f1_pass and recall_pass

    decision = {
        "champion_metrics": champion_metrics,
        "candidate_metrics": candidate_metrics,

        "rules": {
            "candidate_f1_gte_champion_f1": f1_pass,
            "candidate_recall_gte_champion_recall":
                recall_pass
        },

        "decision": (
            "ELIGIBLE_FOR_PROMOTION"
            if eligible
            else "RETAIN_EXISTING_CHAMPION"
        ),

        "champion_overwritten": False,

        "mlflow_status": mlflow_status,

        "metric_justification": (
            "F1 balances precision and recall. "
            "Recall is an additional guardrail because "
            "missing an upcoming engine failure is costly."
        )
    }

    # ==================================================
    # TASK B & D — SAVE REPORTS
    # ==================================================

    with open(
        REPORTS_DIR / "evaluation_report.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump({
            "champion": champion_metrics,
            "candidate": candidate_metrics
        }, file, indent=2)

    with open(
        REPORTS_DIR / "quality_gate_report.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            decision,
            file,
            indent=2
        )

    print("\nTASK D — QUALITY GATE")
    print(json.dumps(decision, indent=2))

    # ==================================================
    # TASK E — GENERATED EVIDENCE
    # ==================================================

    print("\nTASK E — GENERATED EVIDENCE")

    print("data/processed/incoming_batch.csv")
    print("models/champion_model_rebuilt.joblib")
    print("models/candidate_model.joblib")
    print("reports/evaluation_report.json")
    print("reports/quality_gate_report.json")


if __name__ == "__main__":
    main()
