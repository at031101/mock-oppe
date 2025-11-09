##  MLOps OPPE Mock Examination – Completed Pipeline

This repository implements a complete MLOps pipeline covering data versioning, feature store, experiment tracking, and CI/CD.

### ✅ Completed Components
- **DVC (with GCS remote)** – Tracks raw and processed data.
- **Feast Feature Store** – Defines and materializes features locally.
- **MLflow** – Tracks model experiments, parameters, metrics, and artifacts.
- **Hyperparameter Tuning** – Runs small grid search over Logistic Regression `C` values.
- **CI/CD with GitHub Actions** – Pulls data via DVC and tests inference on every commit.
- **Pytest Inference Validation** – Ensures model and preprocessing integrity.

### 🚀 Final Status
| Component | Status |
|------------|---------|
| DVC Remote | ✅ Configured |
| Feast Repo | ✅ Materialized |
| MLflow Tracking | ✅ Working |
| Model Training | ✅ Successful |
| Hyperparameter Search | ✅ Logged |
| GitHub Actions CI | ✅ Passed |

**Final Tag:** `run-v1`
