"""Churn model loading, serving-time preprocessing and prediction.

Preprocessing here must reproduce what the training pipeline
(src/data/preprocess.py + src/features/build_features.py, driven by
scripts/run_pipeline.py) fed the model. tests/test_inference.py checks this by
encoding the raw training CSV with both code paths and asserting the results
are identical.
"""

import os
from collections.abc import Mapping
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

# === MODEL LOCATION ===
# Resolution order:
#   1. $MODEL_DIR, if set (must point at an MLflow model directory)
#   2. /app/model, the path the Dockerfile copies the model to
#   3. the model artifact committed to this repository
MODEL_DIR_ENV_VAR = "MODEL_DIR"
DOCKER_MODEL_DIR = Path("/app/model")
LOCAL_MODEL_DIR = (
    Path(__file__).resolve().parent
    / "model"
    / "2aa2e525c0924aaf83270b1ee3534f53"
    / "artifacts"
)

CHURN_LABEL = "Likely to churn"
NO_CHURN_LABEL = "Not likely to churn"


def _is_mlflow_model_dir(path: Path) -> bool:
    return (path / "MLmodel").is_file()


def resolve_model_dir() -> Path:
    explicit = os.environ.get(MODEL_DIR_ENV_VAR)
    if explicit:
        path = Path(explicit).expanduser()
        if not _is_mlflow_model_dir(path):
            raise FileNotFoundError(
                f"{MODEL_DIR_ENV_VAR}={explicit!r} is not an MLflow model directory "
                f"(no MLmodel file found)."
            )
        return path

    for candidate in (DOCKER_MODEL_DIR, LOCAL_MODEL_DIR):
        if _is_mlflow_model_dir(candidate):
            return candidate

    raise FileNotFoundError(
        "No churn model found. Looked in "
        f"{DOCKER_MODEL_DIR} and {LOCAL_MODEL_DIR}. "
        f"Set {MODEL_DIR_ENV_VAR} to an MLflow model directory."
    )


def _load_model(model_dir: Path):
    # Serving does not use MLflow tracing. Left enabled, loading the model
    # initialises it against MLflow 3.7's default tracking store, which creates
    # mlflow.db and mlruns/ in whatever directory the server was started from.
    mlflow.tracing.disable()
    model = mlflow.pyfunc.load_model(str(model_dir))
    # Some mlflow/dependency combinations (e.g. mlflow 3.7 with SQLAlchemy 2.1)
    # swallow the loading error and return None, so check explicitly.
    if model is None:
        raise RuntimeError(
            f"mlflow.pyfunc.load_model({str(model_dir)!r}) returned None. "
            "Check that installed package versions match requirements.txt."
        )
    return model


def _load_feature_columns(model_dir: Path) -> list[str]:
    feature_file = model_dir / "feature_columns.txt"
    try:
        with open(feature_file) as f:
            cols = [ln.strip() for ln in f if ln.strip()]
    except OSError as e:
        raise FileNotFoundError(f"Failed to load feature columns from {feature_file}: {e}") from e
    if not cols:
        raise ValueError(f"{feature_file} is empty")
    return cols


MODEL_DIR = resolve_model_dir()
model = _load_model(MODEL_DIR)
FEATURE_COLS = _load_feature_columns(MODEL_DIR)
print(f"✅ Loaded churn model and {len(FEATURE_COLS)} feature columns from {MODEL_DIR}")


# === PREPROCESSING SCHEMA ===
# Two-valued columns: build_features._map_binary_series maps Yes/No -> 1/0 and
# Male/Female -> 1/0.
BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

# Multi-class columns: training used pd.get_dummies(drop_first=True) over the
# whole dataset, so categories are sorted and the first one is the dropped
# baseline. Levels are listed here explicitly because a single serving row only
# contains one category per column.
MULTI_CLASS_LEVELS = {
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "No internet service", "Yes"],
    "OnlineBackup": ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "StreamingTV": ["No", "No internet service", "Yes"],
    "StreamingMovies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}

# Numeric columns: preprocess_data coerces TotalCharges and fills numeric NaNs with 0.
NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

RAW_INPUT_COLS = list(BINARY_MAP) + list(MULTI_CLASS_LEVELS) + NUMERIC_COLS


def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Encode raw customer rows into the exact feature matrix the model was trained on.

    Raises ValueError for missing input columns or unknown category values,
    rather than silently encoding them as zeros.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    missing = [c for c in RAW_INPUT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required input fields: {missing}")

    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    for c, mapping in BINARY_MAP.items():
        values = df[c].astype(str).str.strip()
        unknown = sorted(set(values) - set(mapping))
        if unknown:
            raise ValueError(f"Invalid value(s) for {c}: {unknown}. Expected one of {sorted(mapping)}")
        df[c] = values.map(mapping).astype(int)

    for c, levels in MULTI_CLASS_LEVELS.items():
        values = df[c].astype(str).str.strip()
        unknown = sorted(set(values) - set(levels))
        if unknown:
            raise ValueError(f"Invalid value(s) for {c}: {unknown}. Expected one of {levels}")
        # A fixed categorical keeps every level, so drop_first drops the
        # training baseline instead of the only category in a single row.
        df[c] = pd.Categorical(values, categories=levels)

    df = pd.get_dummies(df, columns=list(MULTI_CLASS_LEVELS), drop_first=True, dtype=int)

    not_produced = [c for c in FEATURE_COLS if c not in df.columns]
    if not_produced:
        raise RuntimeError(f"Preprocessing did not produce trained feature columns: {not_produced}")

    # Selecting FEATURE_COLS fixes the column order and drops extras (e.g. customerID).
    return df[FEATURE_COLS]


def _to_label(raw_prediction) -> str:
    values = np.asarray(raw_prediction).ravel()
    if values.size != 1:
        raise RuntimeError(f"Expected exactly one prediction, got {values.size}: {raw_prediction!r}")
    value = values[0]
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise RuntimeError(f"Model returned a non-numeric prediction: {value!r}") from None
    if numeric == 1:
        return CHURN_LABEL
    if numeric == 0:
        return NO_CHURN_LABEL
    raise RuntimeError(f"Model returned unexpected class {value!r}; expected 0 or 1")


def predict(input_dict: Mapping) -> str:
    """Return CHURN_LABEL or NO_CHURN_LABEL for one customer.

    Raises ValueError for invalid input and RuntimeError if the model fails or
    returns something other than a single 0/1 class.
    """
    if not isinstance(input_dict, Mapping):
        raise ValueError(f"Expected a mapping of customer fields, got {type(input_dict).__name__}")

    features = _serve_transform(pd.DataFrame([dict(input_dict)]))

    try:
        raw_prediction = model.predict(features)
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}") from e

    return _to_label(raw_prediction)


if __name__ == "__main__":
    # Example customers
    customers = [
        {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 2,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 89.50,
            "TotalCharges": 179.00
        },
        {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 65,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 95.00,
            "TotalCharges": 6175.00
        },
        {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 70.00,
            "TotalCharges": 70.00
        }
    ]

    print("\n" + "=" * 50)
    print(f"{'Customer':<12} {'Prediction':<25}")
    print("=" * 50)

    for i, cust in enumerate(customers, 1):
        result = predict(cust)
        print(f"Customer {i:<3} {result:<25}")

    print("=" * 50)
