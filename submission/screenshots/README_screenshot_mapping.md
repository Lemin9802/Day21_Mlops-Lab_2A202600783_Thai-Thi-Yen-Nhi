# Screenshot Mapping

## Required screenshots

1. `01_mlflow_ui_3_experiments.png`  
   MLflow UI showing at least 3 runs with parameters and metrics.

2. `02_github_actions_core_pipeline_success.png`  
   GitHub Actions tab showing the core CI/CD jobs in green.

3. `03_curl_health.png`  
   Terminal output from `curl http://VM_IP:8000/health`.

4. `04_curl_predict.png`  
   Terminal output from `curl http://VM_IP:8000/predict`.

5. `05_cloud_storage_data_and_model.png`  
   Google Cloud Storage Console showing DVC data folder and deployed model `models/latest/model.pkl`.

## Bonus screenshots

- `bonus_01_github_gcp_secrets_configured.png`
- `bonus_02_dagshub_remote_tracking.png`
- `bonus_03_github_actions_hyperparameter_tuning_success.png`
- `bonus_04_dagshub_hyperparameter_tuning_runs.png`
- `bonus_05_daily_training_report_artifact.png`
- `bonus_06_regression_gate_pass_log.png`
- `bonus_07_label_distribution_shift_report.png`
