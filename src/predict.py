"""Inference utilities for IMDB Sentiment Analysis."""
import os
import re
import time
import joblib
import pandas as pd
from typing import Dict, Any, List
from src.pipeline import explain

DEFAULT_MODEL_PATH = "models/sentiment_lr.joblib"

# ---------------------------------------------------------------------------
# Phát hiện review "vừa khen vừa chê" bằng cách dự đoán từng vế câu.
# Ví dụ: "Amazing visuals, awful dialogue." -> "Amazing visuals" (positive)
#        + "awful dialogue" (negative) -> các vế trái chiều -> "mixed".
# Chỉ áp dụng cho review ngắn (<= MIXED_MAX_WORDS từ): review IMDB dài thường có
# câu kể nội dung phim mang từ tiêu cực, nếu áp dụng sẽ báo "mixed" oan rất nhiều
# (trên tập test: 8.1% bị gắn mixed so với 7.2% khi chỉ dùng ngưỡng độ tin cậy).
# ---------------------------------------------------------------------------
CLAUSE_SPLIT = re.compile(
    r"(?:<br\s*/?>)|[.;!?,:]+|\b(?:but|however|although|though|whereas|yet)\b",
    re.IGNORECASE,
)
MIXED_MAX_WORDS = 60          # chỉ xét tách vế với review ngắn
CLAUSE_POS_THRESHOLD = 0.6    # vế có P(tích cực) >= 0.6 -> vế khen
CLAUSE_NEG_THRESHOLD = 0.4    # vế có P(tích cực) <= 0.4 -> vế chê
MIN_MINORITY_SHARE = 0.3      # phía ít hơn phải chiếm >= 30% số từ của các vế rõ ràng


def split_clauses(text: str) -> List[str]:
    """Tách review thành các vế theo dấu câu và liên từ đối lập (but, however, although...)."""
    return [p.strip() for p in CLAUSE_SPLIT.split(text) if len(p.strip().split()) >= 2]


def analyze_clauses(text: str, pipe) -> Dict[str, Any]:
    """Dự đoán riêng từng vế câu và cho biết các vế có trái chiều (vừa khen vừa chê) không."""
    if len(text.split()) > MIXED_MAX_WORDS:
        return {"checked": False, "is_mixed": False, "clauses": []}
    clauses = split_clauses(text)
    if len(clauses) < 2:
        return {"checked": True, "is_mixed": False, "clauses": []}
    probs = pipe.predict_proba(pd.DataFrame({"review": clauses}))[:, 1]
    details, pos_words, neg_words = [], 0, 0
    for clause, p in zip(clauses, probs):
        n_words = len(clause.split())
        if p >= CLAUSE_POS_THRESHOLD:
            clause_label = "positive"
            pos_words += n_words
        elif p <= CLAUSE_NEG_THRESHOLD:
            clause_label = "negative"
            neg_words += n_words
        else:
            clause_label = "neutral"
        details.append({"text": clause, "label": clause_label, "confidence": round(float(max(p, 1 - p)), 3)})
    minority_share = min(pos_words, neg_words) / max(pos_words + neg_words, 1)
    is_mixed = pos_words > 0 and neg_words > 0 and minority_share >= MIN_MINORITY_SHARE
    return {"checked": True, "is_mixed": bool(is_mixed), "clauses": details}


def decide_label(text: str, pos_prob: float, pipe, threshold: float = 0.6):
    """Quyết định nhãn cuối cùng: positive / negative / mixed / uncertain.

    - "mixed": review vừa khen vừa chê (các vế câu trái chiều)
    - "uncertain": mô hình không chắc chắn (độ tin cậy < threshold)

    Trả về (label, mixed_reason, clause_info). mixed_reason là:
    - "low_confidence": độ tin cậy của cả câu < threshold
    - "clauses_disagree": có vế khen và vế chê rõ ràng (review ngắn)
    - None: không phải mixed
    """
    conf = max(pos_prob, 1 - pos_prob)
    clause_info = analyze_clauses(text, pipe)
    if conf < threshold:
        return "uncertain", "low_confidence", clause_info
    if clause_info["is_mixed"]:
        return "mixed", "clauses_disagree", clause_info
    return ("positive" if pos_prob >= 0.5 else "negative"), None, clause_info


LABEL_VI = {
    "positive": "Tích cực",
    "negative": "Tiêu cực",
    "mixed": "Vừa khen vừa chê",
    "uncertain": "Không chắc chắn",
}


def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """Load serialized scikit-learn pipeline from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}. Run `python src/train.py` first.")
    return joblib.load(model_path)


def predict_review(
    review: str,
    pipe=None,
    threshold: float = 0.6,
    include_explanation: bool = False,
) -> Dict[str, Any]:
    """Predict sentiment for a single movie review with confidence thresholding."""
    if pipe is None:
        pipe = load_model()
    t0 = time.time()
    X = pd.DataFrame({"review": [review]})
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)[0]
        conf = float(proba.max())
        pos_prob = float(proba[1])
    else:
        pred_cls = int(pipe.predict(X)[0])
        conf = 1.0
        pos_prob = 1.0 if pred_cls == 1 else 0.0

    label, mixed_reason, clause_info = decide_label(review, pos_prob, pipe, threshold)

    latency_ms = round((time.time() - t0) * 1000, 2)
    result = {
        "prediction": label,
        "prediction_vi": LABEL_VI[label],
        "confidence": round(conf, 3),
        "positive_probability": round(pos_prob, 3),
        "latency_ms": latency_ms,
    }
    if clause_info["clauses"]:
        result["clauses"] = clause_info["clauses"]
    if include_explanation and hasattr(pipe.named_steps["model"], "coef_"):
        result["explanation"] = explain(review, pipe=pipe, top=5)
    return result


def predict_batch(reviews: List[str], pipe=None, threshold: float = 0.6) -> List[Dict[str, Any]]:
    """Predict sentiment for a batch of reviews."""
    if pipe is None:
        pipe = load_model()
    t0 = time.time()
    X = pd.DataFrame({"review": reviews})
    probas = pipe.predict_proba(X)
    total_ms = (time.time() - t0) * 1000
    per_item_ms = round(total_ms / max(len(reviews), 1), 3)

    outputs = []
    for text, proba in zip(reviews, probas):
        conf = float(proba.max())
        pos_prob = float(proba[1])
        label, mixed_reason, _ = decide_label(text, pos_prob, pipe, threshold)
        outputs.append({
            "review": text[:100] + ("..." if len(text) > 100 else ""),
            "prediction": label,
            "confidence": round(conf, 3),
            "latency_ms": per_item_ms,
        })
    return outputs

