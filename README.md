# Day 21 MLOps Lab - CI/CD, DVC, MLflow, GCP Deployment

**Student:** Thai Thi Yen Nhi  
**Student ID:** 2A202600783  
**GitHub repository:** https://github.com/Lemin9802/Day21_Mlops-Lab_2A202600783_Thai-Thi-Yen-Nhi  
**DagsHub repository:** https://dagshub.com/Lemin9802/Day21_Mlops-Lab_2A202600783_Thai-Thi-Yen-Nhi  
**Deployed VM IP:** `34.142.132.34`  

This repository contains an end-to-end MLOps lab for a Wine Quality classification task. The project implements data generation, DVC data/pipeline versioning, MLflow/DagsHub experiment tracking, model training/evaluation, GitHub Actions CI/CD, Google Cloud Storage artifact storage, and FastAPI deployment on a Google Cloud VM.

## What is included in this repository

| Area | File or folder | Purpose |
|---|---|---|
| Submission package | [`submission/`](submission/) | Required screenshots and short report for grading |
| Short report | [`submission/REPORT.md`](submission/REPORT.md) | One-page report with selected hyperparameters and difficulties |
| Required screenshots | [`submission/screenshots/`](submission/screenshots/) | Screenshots requested in the submission instructions |
| Extra/bonus evidence | [`submission/screenshots/bonus_and_extra_evidence/`](submission/screenshots/bonus_and_extra_evidence/) | Evidence for DagsHub, tuning, daily report, regression gate, and label shift |
| Data generation | [`generate_data.py`](generate_data.py), [`add_new_data.py`](add_new_data.py) | Generate Wine Quality train/eval datasets and simulate new data |
| Training | [`src/train.py`](src/train.py) | Train RandomForest model and log metrics/artifacts to MLflow |
| Evaluation | [`src/evaluate.py`](src/evaluate.py) | Evaluate model and write `outputs/metrics.json` |
| Serving | [`src/serve.py`](src/serve.py) | FastAPI service with `/health` and `/predict` endpoints |
| Hyperparameter tuning | [`src/tune.py`](src/tune.py) | Run multiple algorithms and log tuning runs to MLflow/DagsHub |
| Daily training report | [`src/report.py`](src/report.py) | Generate Markdown training report artifact |
| Label shift check | [`src/check_label_shift.py`](src/check_label_shift.py) | Compare train/eval label distribution and create shift report |
| Tests | [`tests/test_train.py`](tests/test_train.py) | Unit tests for training output, metrics, and model artifact |
| DVC pipeline | [`dvc.yaml`](dvc.yaml), [`dvc.lock`](dvc.lock) | Reproducible `prepare -> train -> evaluate` pipeline |
| CI/CD workflow | [`.github/workflows/mlops.yml`](.github/workflows/mlops.yml) | GitHub Actions pipeline for test, tune, train, eval gate, deploy |
| Hyperparameters | [`params.yaml`](params.yaml) | Final selected model hyperparameters |
| Regression baseline | [`config/baseline_metrics.json`](config/baseline_metrics.json) | Baseline accuracy used to block deploy on regression |
| Dependencies | [`requirements.txt`](requirements.txt) | Python dependencies |

## Submission checklist

The required submission evidence is inside [`submission/screenshots/`](submission/screenshots/):

| Requirement | Screenshot |
|---|---|
| MLflow UI showing at least 3 experiments/runs | [`01_mlflow_ui_3_experiments.png`](submission/screenshots/01_mlflow_ui_3_experiments.png) |
| GitHub Actions tab showing green jobs | [`02_github_actions_core_pipeline_success.png`](submission/screenshots/02_github_actions_core_pipeline_success.png) |
| `curl http://VM_IP:8000/health` result | [`03_curl_health.png`](submission/screenshots/03_curl_health.png) |
| `curl http://VM_IP:8000/predict` result | [`04_curl_predict.png`](submission/screenshots/04_curl_predict.png) |
| Cloud Storage Console showing DVC data and deployed model | [`05_cloud_storage_data_and_model.png`](submission/screenshots/05_cloud_storage_data_and_model.png) |

The short report is here:

- [`submission/REPORT.md`](submission/REPORT.md)

## Pipeline overview

The project pipeline is:

```text
Generate data -> DVC repro -> Train model -> Evaluate model -> Gate check -> Upload model to GCS -> Deploy FastAPI service
```

GitHub Actions workflow:

```text
Unit Test -> Hyperparameter Tuning -> Train -> Eval Gate -> Deploy to VM
```

The workflow also creates these artifacts:

```text
daily-training-report
tuning-results
```

## Model selection

The final deployed model is `RandomForestClassifier` with:

```yaml
n_estimators: 200
max_depth: 10
min_samples_split: 2
```

This was selected after hyperparameter tuning across:

```text
RandomForestClassifier
GradientBoostingClassifier
LogisticRegression
```

Best tuning result:

```text
accuracy = 0.6480
weighted_f1_score = 0.6464
```

## MLflow and DagsHub tracking

Local MLflow was used for the initial 3 experiments. DagsHub was then configured as the remote MLflow tracking server.

- DagsHub repo: https://dagshub.com/Lemin9802/Day21_Mlops-Lab_2A202600783_Thai-Thi-Yen-Nhi
- Evidence: [`submission/screenshots/bonus_and_extra_evidence/bonus_02_dagshub_remote_tracking.png`](submission/screenshots/bonus_and_extra_evidence/bonus_02_dagshub_remote_tracking.png)
- Hyperparameter tuning evidence: [`submission/screenshots/bonus_and_extra_evidence/bonus_04_dagshub_hyperparameter_tuning_runs.png`](submission/screenshots/bonus_and_extra_evidence/bonus_04_dagshub_hyperparameter_tuning_runs.png)

## DVC and Cloud Storage

DVC remote storage is configured with Google Cloud Storage:

```text
gs://day21-mlops-2a202600783-forward-alchemy/dvc
```

The deployed model is uploaded to:

```text
gs://day21-mlops-2a202600783-forward-alchemy/models/latest/model.pkl
```

Cloud Storage evidence:

- [`submission/screenshots/05_cloud_storage_data_and_model.png`](submission/screenshots/05_cloud_storage_data_and_model.png)

## Deployed API

FastAPI is deployed on a Google Cloud VM.

Health endpoint:

```text
GET http://34.142.132.34:8000/health
```

Prediction endpoint:

```text
POST http://34.142.132.34:8000/predict
```

Example request:

```powershell
@'
{"features":[7.4,0.7,0.0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4,0]}
'@ | Set-Content predict_body.json -Encoding utf8

curl.exe -X POST "http://34.142.132.34:8000/predict" `
  -H "Content-Type: application/json" `
  --data-binary "@predict_body.json"
```

## Bonus tasks completed

| Bonus | Implementation | Evidence |
|---|---|---|
| DagsHub remote experiment tracking | GitHub Actions logs MLflow runs to DagsHub | [`bonus_02_dagshub_remote_tracking.png`](submission/screenshots/bonus_and_extra_evidence/bonus_02_dagshub_remote_tracking.png) |
| Multiple algorithms and hyperparameter tuning | `src/tune.py` runs RandomForest, GradientBoosting, LogisticRegression | [`bonus_04_dagshub_hyperparameter_tuning_runs.png`](submission/screenshots/bonus_and_extra_evidence/bonus_04_dagshub_hyperparameter_tuning_runs.png) |
| Daily training report | `src/report.py` creates `daily_training_report.md` artifact | [`bonus_05_daily_training_report_artifact.png`](submission/screenshots/bonus_and_extra_evidence/bonus_05_daily_training_report_artifact.png) |
| Block deploy on regression | Eval Gate compares current accuracy against baseline | [`bonus_06_regression_gate_pass_log.png`](submission/screenshots/bonus_and_extra_evidence/bonus_06_regression_gate_pass_log.png) |
| Label distribution shift report | `src/check_label_shift.py` creates label distribution report | [`bonus_07_label_distribution_shift_report.png`](submission/screenshots/bonus_and_extra_evidence/bonus_07_label_distribution_shift_report.png) |

## Reproduce locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python generate_data.py
dvc repro
pytest tests/ -v
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
python generate_data.py
dvc repro
pytest tests/ -v
```
