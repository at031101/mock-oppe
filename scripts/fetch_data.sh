#!/usr/bin/env bash
set -e

# Generate the Iris dataset and save as CSV
python - <<'PY'
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df = pd.DataFrame(iris.data, columns=["sepal_length","sepal_width","petal_length","petal_width"])
df["species"] = [iris.target_names[i] for i in iris.target]
df["sample_id"] = range(1, len(df)+1)

import os
os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/iris_v0.csv", index=False)
print("✅ Wrote data/raw/iris_v0.csv")
PY
