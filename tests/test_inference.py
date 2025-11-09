import os
import joblib
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

def preprocess(df, target_col="target"):
    """Encode any categorical columns (similar to training preprocessing)."""
    if target_col in df.columns:
        df = df.drop(columns=[target_col])
    for col in df.columns:
        if df[col].dtype == "object" or str(df[col].dtype).startswith("category"):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df

@pytest.mark.parametrize("model_path", [
    "models/logreg_C_1.0.joblib",
    "models/logreg_C_0.1.joblib",
])
def test_model_inference(model_path):
    """Ensure model loads, preprocesses input correctly, and predicts."""
    if not os.path.exists(model_path):
        pytest.skip(f"{model_path} not found; skipping inference test")

    model = joblib.load(model_path)
    data_path = "data/processed/val.parquet"
    assert os.path.exists(data_path), f"Validation data {data_path} missing"

    df = pd.read_parquet(data_path)
    X = preprocess(df, target_col="target")

    preds = model.predict(X.head(5))
    assert len(preds) == 5, "Prediction output length mismatch"
    print(f"✅ {os.path.basename(model_path)} predicted 5 samples successfully.")
