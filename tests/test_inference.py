"""Tests for the real model artifact and serving-time preprocessing (no mocks)."""

from pathlib import Path

import pandas as pd
import pytest

from src.app.main import HIGH_RISK_EXAMPLE, INPUT_FIELDS, LOW_RISK_EXAMPLE
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.serving import inference

VALID_LABELS = {inference.CHURN_LABEL, inference.NO_CHURN_LABEL}
RAW_DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "Telco-Churn.csv"


def _ui_example_payload(example):
    data = dict(zip(INPUT_FIELDS, example, strict=True))
    data["SeniorCitizen"] = 1 if data["SeniorCitizen"] == "Yes" else 0
    return data


HIGH_RISK = _ui_example_payload(HIGH_RISK_EXAMPLE)
LOW_RISK = _ui_example_payload(LOW_RISK_EXAMPLE)


@pytest.fixture(scope="module")
def training_encoded():
    """Raw CSV encoded exactly as scripts/run_pipeline.py does before training."""
    raw = pd.read_csv(RAW_DATA)
    df = build_features(preprocess_data(raw.copy()), target_col="Churn")
    for c in df.select_dtypes(include=["bool"]).columns:
        df[c] = df[c].astype(int)
    return raw, df.drop(columns=["Churn"])


# --- model loading ---

def test_model_artifact_loads():
    assert inference.model is not None
    assert (inference.MODEL_DIR / "MLmodel").is_file()
    assert len(inference.FEATURE_COLS) == 30


def test_feature_columns_match_trained_booster():
    booster = inference.model.get_raw_model().get_booster()
    assert booster.feature_names == inference.FEATURE_COLS


def test_resolve_prefers_local_artifact_outside_docker(monkeypatch):
    monkeypatch.delenv(inference.MODEL_DIR_ENV_VAR, raising=False)
    if inference.DOCKER_MODEL_DIR.exists():
        pytest.skip("running inside the Docker image")
    assert inference.resolve_model_dir() == inference.LOCAL_MODEL_DIR


def test_resolve_honours_env_var(monkeypatch):
    monkeypatch.setenv(inference.MODEL_DIR_ENV_VAR, str(inference.LOCAL_MODEL_DIR))
    assert inference.resolve_model_dir() == inference.LOCAL_MODEL_DIR


def test_resolve_rejects_invalid_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv(inference.MODEL_DIR_ENV_VAR, str(tmp_path))
    with pytest.raises(FileNotFoundError, match="not an MLflow model directory"):
        inference.resolve_model_dir()


# --- predictions on the UI examples ---

@pytest.mark.parametrize("customer", [HIGH_RISK, LOW_RISK], ids=["high_risk", "low_risk"])
def test_example_customers_return_valid_label(customer):
    result = inference.predict(customer)
    assert result is not None
    assert isinstance(result, str)
    assert result in VALID_LABELS


def test_both_classes_map_to_a_label():
    # Guards against the old unreachable `return` that made class 0 return None.
    assert inference._to_label([1]) == inference.CHURN_LABEL
    assert inference._to_label([0]) == inference.NO_CHURN_LABEL


@pytest.mark.parametrize("bad", [[], [0, 1], ["maybe"], [2]])
def test_unexpected_model_output_raises(bad):
    with pytest.raises(RuntimeError):
        inference._to_label(bad)


# --- preprocessing ---

def test_serve_transform_produces_exact_trained_columns():
    enc = inference._serve_transform(pd.DataFrame([HIGH_RISK]))
    assert list(enc.columns) == inference.FEATURE_COLS
    assert len(enc) == 1


def test_single_row_one_hot_is_not_dropped():
    enc = inference._serve_transform(pd.DataFrame([HIGH_RISK])).iloc[0]
    assert enc["InternetService_Fiber optic"] == 1
    assert enc["PaymentMethod_Electronic check"] == 1
    assert enc["StreamingTV_Yes"] == 1
    assert enc["Contract_One year"] == 0 and enc["Contract_Two year"] == 0


def test_schema_levels_match_training_data():
    raw = pd.read_csv(RAW_DATA)
    for col, mapping in inference.BINARY_MAP.items():
        assert set(raw[col].unique()) == set(mapping), col
    for col, levels in inference.MULTI_CLASS_LEVELS.items():
        assert sorted(raw[col].unique()) == levels, col


def test_serving_matches_training_encoding_full_dataset(training_encoded):
    raw, expected = training_encoded
    served = inference._serve_transform(raw.drop(columns=["Churn"]))
    pd.testing.assert_frame_equal(
        served.reset_index(drop=True).astype(float),
        expected.reset_index(drop=True).astype(float),
    )


def test_serving_matches_training_encoding_row_by_row(training_encoded):
    # The API encodes one row at a time; that is where drop_first used to break.
    raw, expected = training_encoded
    sample = raw.drop(columns=["Churn"]).sample(200, random_state=0)
    for idx, row in sample.iterrows():
        served = inference._serve_transform(row.to_frame().T)
        assert (served.iloc[0].astype(float).values == expected.loc[idx].astype(float).values).all(), idx


def test_missing_field_is_rejected():
    customer = dict(LOW_RISK)
    del customer["Contract"]
    with pytest.raises(ValueError, match="Missing required input fields"):
        inference.predict(customer)


def test_unknown_category_is_rejected():
    with pytest.raises(ValueError, match="Invalid value"):
        inference.predict({**LOW_RISK, "Contract": "Three year"})
