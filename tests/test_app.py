"""Tests for the FastAPI app and the Gradio callback, using the real model."""

import pytest
from fastapi.testclient import TestClient

from src.app.main import HIGH_RISK_EXAMPLE, INPUT_FIELDS, LOW_RISK_EXAMPLE, app, gradio_predict
from src.serving.inference import CHURN_LABEL, NO_CHURN_LABEL

VALID_LABELS = {CHURN_LABEL, NO_CHURN_LABEL}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _api_payload(example):
    data = dict(zip(INPUT_FIELDS, example, strict=True))
    data["SeniorCitizen"] = 1 if data["SeniorCitizen"] == "Yes" else 0
    return data


def test_root_returns_status_ok(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ui_is_mounted(client):
    assert client.get("/ui/").status_code == 200


@pytest.mark.parametrize("example", [HIGH_RISK_EXAMPLE, LOW_RISK_EXAMPLE], ids=["high_risk", "low_risk"])
def test_predict_endpoint_with_valid_payload(client, example):
    resp = client.post("/predict", json=_api_payload(example))
    assert resp.status_code == 200
    assert resp.json()["prediction"] in VALID_LABELS


def test_predict_endpoint_senior_citizen_is_optional(client):
    payload = _api_payload(LOW_RISK_EXAMPLE)
    del payload["SeniorCitizen"]
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    assert resp.json()["prediction"] in VALID_LABELS


def test_predict_endpoint_rejects_unknown_category(client):
    resp = client.post("/predict", json={**_api_payload(LOW_RISK_EXAMPLE), "Contract": "Three year"})
    assert resp.status_code == 422
    assert "Contract" in resp.json()["detail"]


@pytest.mark.parametrize("example", [HIGH_RISK_EXAMPLE, LOW_RISK_EXAMPLE], ids=["high_risk", "low_risk"])
def test_gradio_examples_end_to_end(example):
    assert gradio_predict(*example) in VALID_LABELS


def test_gradio_predict_reports_bad_input_instead_of_crashing():
    values = list(LOW_RISK_EXAMPLE)
    values[INPUT_FIELDS.index("tenure")] = None  # empty Number field
    assert gradio_predict(*values).startswith("Invalid input:")


def test_deprecated_app_module_is_the_canonical_app():
    from src.app import app as legacy

    assert legacy.app is app
