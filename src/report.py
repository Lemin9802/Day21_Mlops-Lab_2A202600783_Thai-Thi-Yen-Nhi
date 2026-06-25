import json
import os
from datetime import datetime, timezone


METRICS_PATH = "outputs/metrics.json"
REPORT_PATH = "outputs/daily_training_report.md"


def main() -> None:
    """
    Tạo daily training report dạng Markdown để lưu làm GitHub Actions artifact.

    Report này giúp reviewer thấy rõ:
    - commit nào đã train model
    - metrics hiện tại
    - eval gate pass/fail
    - model artifact đang được deploy ở đâu
    """
    os.makedirs("outputs", exist_ok=True)

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    accuracy = float(metrics["accuracy"])
    f1_score = float(metrics["f1_score"])
    threshold = float(metrics.get("eval_threshold", 0.60))
    passed = accuracy >= threshold

    commit_sha = os.environ.get("GITHUB_SHA", "local-run")
    run_id = os.environ.get("GITHUB_RUN_ID", "local-run")
    gcs_bucket = os.environ.get("GCS_BUCKET", "day21-mlops-2a202600783-forward-alchemy")
    gcs_model_key = os.environ.get("GCS_MODEL_KEY", "models/latest/model.pkl")

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    report = f"""# Daily Training Report

## Run metadata

- Generated at: {generated_at}
- Commit SHA: `{commit_sha}`
- GitHub Actions run ID: `{run_id}`
- Model artifact: `gs://{gcs_bucket}/{gcs_model_key}`

## Evaluation metrics

| Metric | Value |
|---|---:|
| Accuracy | {accuracy:.4f} |
| Weighted F1-score | {f1_score:.4f} |
| Eval threshold | {threshold:.4f} |
| Gate passed | {passed} |

## Deployment decision

"""

    if passed:
        report += (
            "The model passed the evaluation gate and is eligible for deployment.\n"
        )
    else:
        report += (
            "The model failed the evaluation gate and should not be deployed.\n"
        )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Daily training report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()