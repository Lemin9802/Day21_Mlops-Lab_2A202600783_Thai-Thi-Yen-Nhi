# Day 21 - MLOps Lab: CI/CD for AI Systems

Student: Thai Thi Yen Nhi
Course: AIInAction - VinUni
Topic: Day 21 - CI/CD for AI Systems

## Lab Goals

This lab builds an end-to-end MLOps workflow:

1. Track local ML experiments with MLflow.
2. Version datasets with DVC and a cloud object storage remote.
3. Run a GitHub Actions pipeline with unit test, train, eval gate, and deploy jobs.
4. Serve the trained model as a FastAPI REST API on a cloud VM.
5. Simulate continuous training when new data is added.

## Dataset

The lab uses the UCI Wine Quality dataset. The prediction target has three classes:

- 0: low quality
- 1: medium quality
- 2: high quality

After running `python generate_data.py`, the local data folder should contain:

- `data/train_phase1.csv`
- `data/eval.csv`
- `data/train_phase2.csv`

## Main Files

- `generate_data.py`: download and split the dataset.
- `add_new_data.py`: append phase 2 data to phase 1 data.
- `params.yaml`: model hyperparameters.
- `src/train.py`: training script with MLflow logging.
- `src/serve.py`: FastAPI inference server.
- `tests/test_train.py`: unit tests.
- `.github/workflows/mlops.yml`: CI/CD workflow skeleton.

## Submission Evidence

- MLflow UI screenshot with at least 3 runs.
- GitHub Actions screenshot with all jobs passing.
- Cloud storage screenshot showing DVC data and uploaded model.
- `curl /health` and `curl /predict` outputs from the deployed VM.
- Short report explaining the best hyperparameters and issues encountered.
