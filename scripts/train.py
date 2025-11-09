#!/usr/bin/env python3
"""
Simple Logistic Regression training script.
Automatically encodes categorical columns before training.
Logs metrics and model to MLflow.
"""

import os
import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import joblib

def main(args):
    # connect to MLflow tracking server
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", f"http://127.0.0.1:{args.mlflow_port}")
    mlflow.set_tracking_uri(mlflow_uri)
    print(f"MLflow tracking URI: {mlflow_uri}")

    # load dataset
    df = pd.read_parquet(args.data_file)
    print(f"Loaded data shape: {df.shape}")

    # target and features
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in dataset!")

    y = df[args.target]
    X = df.drop(columns=[args.target])

    # encode any non-numeric columns
    for col in X.columns:
        if X[col].dtype == "object" or str(X[col].dtype).startswith("category"):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            print(f"Encoded column '{col}' with LabelEncoder.")

    # split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # evaluate
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"✅ Validation Accuracy: {acc:.4f}")

    # log to MLflow
    with mlflow.start_run(run_name="logreg_iris_fixed"):
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_metric("val_accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

    # save locally too
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "logreg_model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved at {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", default="data/processed/train.parquet")
    parser.add_argument("--target", default="target")
    parser.add_argument("--mlflow-port", type=int, default=5000)
    args = parser.parse_args()
    main(args)
