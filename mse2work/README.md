# NASA FD001 Model Quality Monitoring and Retraining

This project implements the five assignment tasks using the **official NASA C-MAPSS FD001 dataset**. It does not generate substitute or synthetic data.

## Dataset setup

Copy these three official files into `data/raw/`:

- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

They are available from the NASA Prognostics Data Repository. The incoming labelled batch is derived from `test_FD001.txt` and its matching `RUL_FD001.txt` labels. The final RUL label is added to the difference between each engine's final test cycle and each earlier cycle.

## Install and run

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.monitoring
```

## Assignment mapping

- **Task A:** Load incoming NASA test data, derive labels, validate required columns, missing values and duplicate engine-cycle records, then report row count and target distribution.
- **Task B:** Evaluate the champion classifier on a held-out set of incoming engines using Accuracy, Precision, Recall and F1. Save `reports/evaluation_report.json`.
- **Task C:** Combine original NASA training data with the incoming training-engine subset, train a Random Forest candidate using the same 25 feature columns, and save `models/candidate_model.joblib`. MLflow tracking is attempted using a local SQLite tracking store when installed.
- **Task D:** Candidate is eligible only when F1 is no worse and Recall is no worse than the champion. The champion is never overwritten automatically. The decision is saved in `reports/quality_gate_report.json`.
- **Task E:** Console output and the generated CSV, model and JSON files provide execution evidence.

## Important model-artifact note

Scikit-learn/joblib model files are not guaranteed to be portable across Python and scikit-learn versions. To avoid silently using an incompatible artifact, this workflow rebuilds a reproducible champion from the original NASA training data with the project's Random Forest configuration and saves it as `models/champion_model_rebuilt.joblib`. It does not overwrite `models/failure_classifier.joblib`.

## Metric justification

F1 is the primary quality metric because it balances Precision and Recall. Recall is an additional guardrail because a false negative means failing to flag an engine approaching failure.
