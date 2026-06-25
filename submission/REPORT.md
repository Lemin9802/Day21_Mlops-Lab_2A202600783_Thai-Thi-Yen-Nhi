# Short Report

## Selected hyperparameters and reason

The final deployed model uses `RandomForestClassifier` with:

```yaml
n_estimators: 200
max_depth: 10
min_samples_split: 2
```

This configuration was selected after running hyperparameter tuning across three algorithms: RandomForest, GradientBoosting, and LogisticRegression. The best tuning result was the RandomForest run with `n_estimators=200`, `max_depth=10`, and `min_samples_split=2`, which achieved `accuracy = 0.6480` and `weighted F1-score = 0.6464`. This was better than the previous stable baseline accuracy of `0.6440`, so the model passed both the evaluation gate and the regression gate before deployment.

## Difficulties and solutions

One issue was an MLflow import error related to `pkg_resources`. I fixed this by adding `setuptools<81` to `requirements.txt`. Another issue was that DVC on Windows sometimes used the global Python interpreter instead of the project virtual environment, which caused missing package errors. I solved this by adding `run_stage.py`, so DVC stages run with the local `.venv` interpreter. During cloud setup, creating a new GCP project failed due to project quota limits, so I reused the existing project `forward-alchemy-442913-f9`, linked billing, created a GCS bucket, and configured DVC remote storage. During API testing, PowerShell quoting caused invalid JSON for `/predict`, so I wrote the request body to `predict_body.json` and sent it with `curl --data-binary`.

The final pipeline successfully runs unit tests, hyperparameter tuning, training, evaluation gate, regression gate, model upload to Google Cloud Storage, and deployment to a Google Cloud VM running FastAPI.
