"""Unit tests for IMDB Sentiment Analysis pipeline, feature engineering, serialization, and FastAPI."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from src.features import clean_text, MetaFeatures
from src.pipeline import build_pipeline, explain
from src.monitor import check_data_drift
from api.app import app


def test_clean_text_removes_html_and_preserves_negations():
    raw = "This movie was <br /><br /> NOT good at all!   Worst film ever."
    cleaned = clean_text(raw)
    assert "<br" not in cleaned
    assert "not good" in cleaned
    assert "  " not in cleaned


def test_meta_features_shape_and_values():
    reviews = pd.Series([
        "WOW!! This is NOT bad!",
        "Simple short review.",
    ])
    transformer = MetaFeatures()
    out = transformer.fit_transform(reviews)
    assert out.shape == (2, 4)
    # First review: 5 words, 3 '!' marks, >0 uppercase ratio, 1 negation ('NOT')
    assert out[0, 0] == 5
    assert out[0, 1] == 3
    assert out[0, 2] > 0.2
    assert out[0, 3] == 1
    # Second review: 3 words, 0 '!', 0 negations
    assert out[1, 0] == 3
    assert out[1, 1] == 0
    assert out[1, 3] == 0


def test_pipeline_fit_predict_and_serialization():
    sample_df = pd.DataFrame({
        "review": [
            "An absolute masterpiece, brilliant acting and a beautiful story!",
            "Great film, wonderful direction and superb cast.",
            "Loved every minute of this amazing cinema experience!",
            "Fantastic screenplay, deeply moving and entertaining.",
            "Brilliant cinematography and unforgettable performances.",
            "Worst movie I have ever seen, a total waste of time.",
            "Terrible plot, boring dialogue and awful acting!",
            "Horrible direction, completely unwatchable garbage.",
            "Awful script, dull characters and a painful ending.",
            "Disappointing and tedious waste of two hours.",
        ] * 3
    })
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0] * 3)

    pipe = build_pipeline(LogisticRegression(random_state=42), min_df=1)
    pipe.fit(sample_df, y)
    preds_before = pipe.predict_proba(sample_df)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_file = os.path.join(tmpdir, "test_model.joblib")
        joblib.dump(pipe, model_file)
        loaded_pipe = joblib.load(model_file)
        preds_after = loaded_pipe.predict_proba(sample_df)

    np.testing.assert_allclose(preds_before, preds_after, rtol=1e-6)


def test_explain_function_outputs_words():
    sample_df = pd.DataFrame({
        "review": [
            "great acting wonderful story masterpiece",
            "great film brilliant cast amazing",
            "terrible plot boring movie awful",
            "worst waste of time boring terrible",
        ] * 4
    })
    y = np.array([1, 1, 0, 0] * 4)
    pipe = build_pipeline(LogisticRegression(random_state=42), min_df=1)
    pipe.fit(sample_df, y)

    res = explain("The acting was great but the plot was terrible and boring", pipe=pipe, top=3)
    assert "negative_words" in res and "positive_words" in res


def test_data_drift_monitor():
    ref = pd.Series(["Short positive review."] * 50)
    cur = pd.Series(["VERY LONG EXTREME REVIEW!!! NOT GOOD AT ALL!!! " * 10] * 50)
    report = check_data_drift(ref, cur)
    assert report["overall_drift_detected"] is True
    assert "word_count" in report["features"]


def test_api_health_and_model_info():
    client = TestClient(app)
    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    r_info = client.get("/model-info")
    assert r_info.status_code == 200
    assert "model_name" in r_info.json()


def test_api_input_validation_rejects_empty_review():
    client = TestClient(app)
    r_empty = client.post("/predict", json={"text": ""})
    assert r_empty.status_code == 422

    r_whitespace = client.post("/predict", json={"text": "   "})
    assert r_whitespace.status_code == 422


def test_api_predict_and_batch_predict_when_model_exists():
    if not os.path.exists("models/sentiment_lr.joblib"):
        pytest.skip("Trained model artifact not generated yet")
    client = TestClient(app)

    r1 = client.post("/predict", json={"text": "An absolute masterpiece, brilliant acting and a beautiful story"})
    assert r1.status_code == 200
    data1 = r1.json()
    assert data1["prediction"] in {"positive", "negative", "mixed", "uncertain"}
    assert 0.0 <= data1["confidence"] <= 1.0

    # Also test rubric format {"features": ["..."]}
    r2 = client.post("/predict", json={"features": ["Worst movie I have ever seen, a total waste of time"]})
    assert r2.status_code == 200
    assert r2.json()["prediction"] == "negative"

    # Batch predict
    r_batch = client.post(
        "/batch-predict",
        json={"instances": ["An absolute masterpiece!", "Worst movie ever, total waste of time."]},
    )
    assert r_batch.status_code == 200
    assert len(r_batch.json()["predictions"]) == 2


def test_mixed_review_detected_by_clauses():
    """Review ngắn có một vế khen rõ và một vế chê rõ phải được gắn nhãn mixed."""
    if not os.path.exists("models/sentiment_lr.joblib"):
        pytest.skip("Trained model artifact not generated yet")
    from src.predict import load_model, predict_review, split_clauses

    assert split_clauses("Amazing visuals, awful dialogue.") == ["Amazing visuals", "awful dialogue"]
    pipe = load_model()
    mixed = predict_review("Amazing visuals, awful dialogue.", pipe=pipe)
    assert mixed["prediction"] == "mixed"
    assert [c["label"] for c in mixed["clauses"]] == ["positive", "negative"]
    clear = predict_review("Worst movie I have ever seen, a total waste of time", pipe=pipe)
    assert clear["prediction"] == "negative"
