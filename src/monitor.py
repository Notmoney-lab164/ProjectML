"""Model monitoring and data drift detection utilities."""
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.features import MetaFeatures


def check_data_drift(
    reference_reviews: pd.Series,
    current_reviews: pd.Series,
    alpha: float = 0.05,
) -> dict:
    """Detect covariate shift / data drift on review meta-features using Two-Sample Kolmogorov-Smirnov test."""
    extractor = MetaFeatures()
    ref_meta = extractor.transform(reference_reviews)
    cur_meta = extractor.transform(current_reviews)
    feature_names = extractor.get_feature_names_out()

    drift_report = {}
    any_drift = False
    for idx, fname in enumerate(feature_names):
        stat, p_val = ks_2samp(ref_meta[:, idx], cur_meta[:, idx])
        drifted = bool(p_val < alpha)
        if drifted:
            any_drift = True
        drift_report[str(fname)] = {
            "ks_statistic": round(float(stat), 4),
            "p_value": round(float(p_val), 4),
            "drift_detected": drifted,
            "ref_mean": round(float(np.mean(ref_meta[:, idx])), 3),
            "cur_mean": round(float(np.mean(cur_meta[:, idx])), 3),
        }

    return {
        "overall_drift_detected": any_drift,
        "alpha": alpha,
        "reference_samples": int(len(reference_reviews)),
        "current_samples": int(len(current_reviews)),
        "features": drift_report,
    }


def analyze_prediction_logs(log_path: str = "logs/predictions.csv") -> dict:
    """Summarize production API prediction logs for latency, confidence decay, and class distribution."""
    if not os.path.exists(log_path):
        return {"status": "no_logs_found", "total_requests": 0}

    df = pd.read_csv(log_path)
    if df.empty:
        return {"status": "empty_logs", "total_requests": 0}

    low_conf_rate = float((df["confidence"] < 0.6).mean())
    return {
        "status": "ok",
        "total_requests": int(len(df)),
        "prediction_distribution": df["prediction"].value_counts().to_dict(),
        "mean_confidence": round(float(df["confidence"].mean()), 4),
        "low_confidence_rate": round(low_conf_rate, 4),
        "mean_latency_ms": round(float(df["latency_ms"].mean()), 2),
        "p95_latency_ms": round(float(df["latency_ms"].quantile(0.95)), 2),
        "alert_low_confidence": bool(low_conf_rate > 0.25),
    }


if __name__ == "__main__":
    if os.path.exists("data/train.csv") and os.path.exists("data/test.csv"):
        train_df = pd.read_csv("data/train.csv").sample(2000, random_state=42)
        test_df = pd.read_csv("data/test.csv").sample(1000, random_state=42)
        report = check_data_drift(train_df["review"], test_df["review"])
        os.makedirs("reports", exist_ok=True)
        with open("reports/monitoring_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print("Monitoring drift report saved to reports/monitoring_report.json:")
        print(json.dumps(report, indent=2))

