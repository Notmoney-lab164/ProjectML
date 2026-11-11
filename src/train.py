"""End-to-end training script for IMDB Sentiment Analysis with MLflow tracking."""
import os
import sys
import json
import time
from datetime import datetime

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_collection import download_imdb_dataset
from src.pipeline import build_pipeline


def run_training():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # 1. Load & deduplicate dataset
    df = download_imdb_dataset()
    raw_len = len(df)
    dup_count = int(df.duplicated("review").sum())
    html_count = int(df["review"].str.contains("<br", na=False).sum())
    df = df.drop_duplicates("review").reset_index(drop=True)
    df["label"] = (df["sentiment"] == "positive").astype(int)
    print(f"Raw dataset: {raw_len} rows | Duplicates removed: {dup_count} | Remaining: {len(df)} | HTML <br /> count: {html_count}")

    X, y = df[["review"]], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save train/test splits for reproducibility & DistilBERT
    train_out = X_train.assign(label=y_train)
    test_out = X_test.assign(label=y_test)
    train_out.to_csv("data/train.csv", index=False)
    test_out.to_csv("data/test.csv", index=False)
    train_out.to_csv("data/processed/train.csv", index=False)
    test_out.to_csv("data/processed/test.csv", index=False)
    print(f"Saved splits: Train={len(X_train)}, Test={len(X_test)}")

    # Configure MLflow local tracking in ./mlruns
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    os.makedirs("mlruns", exist_ok=True)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("IMDB_Sentiment_Analysis")

    # 2. Train 3 classical models
    models = {
        "Logistic Regression (Baseline)": (LogisticRegression(max_iter=1000, random_state=42), None, False),
        "Linear SVM": (LinearSVC(random_state=42), None, False),
        "Random Forest": (RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42), 20000, False),
    }

    results = []
    pipes = {}

    for name, (model, max_feat, tuned) in models.items():
        print(f"\nTraining {name}...")
        with mlflow.start_run(run_name=name):
            pipe = build_pipeline(model, max_features=max_feat)
            t0 = time.time()
            pipe.fit(X_train, y_train)
            train_time = time.time() - t0

            t0 = time.time()
            pred_test = pipe.predict(X_test)
            infer_ms = (time.time() - t0) / len(X_test) * 1000
            pred_train = pipe.predict(X_train)

            # 5-fold CV score on a stratified subset if RF to keep runtime fast, or full for linear models
            cv_scores = cross_val_score(
                pipe,
                X_train.iloc[:10000] if max_feat else X_train,
                y_train.iloc[:10000] if max_feat else y_train,
                cv=5,
                scoring="f1",
                n_jobs=-1,
            )
            cv_f1 = float(np.mean(cv_scores))

            tr_f1 = float(f1_score(y_train, pred_train))
            te_f1 = float(f1_score(y_test, pred_test))
            te_acc = float(accuracy_score(y_test, pred_test))
            te_prec = float(precision_score(y_test, pred_test))
            te_rec = float(recall_score(y_test, pred_test))

            mlflow.log_param("model_name", name)
            mlflow.log_param("max_features", str(max_feat))
            mlflow.log_param("ngram_range", "(1, 2)")
            mlflow.log_param("random_state", 42)
            mlflow.log_metric("train_f1", round(tr_f1, 4))
            mlflow.log_metric("test_f1", round(te_f1, 4))
            mlflow.log_metric("cv_f1", round(cv_f1, 4))
            mlflow.log_metric("test_accuracy", round(te_acc, 4))
            mlflow.log_metric("train_time_s", round(train_time, 2))
            mlflow.log_metric("infer_ms_per_review", round(infer_ms, 4))

            pipes[name] = pipe
            row = {
                "Model": name,
                "Train F1": round(tr_f1, 4),
                "Test F1": round(te_f1, 4),
                "CV F1 (5-fold)": round(cv_f1, 4),
                "Test Acc": round(te_acc, 4),
                "Test Precision": round(te_prec, 4),
                "Test Recall": round(te_rec, 4),
                "Train time (s)": round(train_time, 1),
                "ms / review": round(infer_ms, 3),
                "Tuned": "No",
            }
            results.append(row)
            print(row)

    # 3. Hyperparameter tuning for Logistic Regression via 5-fold GridSearchCV
    print("\nRunning 5-fold GridSearchCV on Logistic Regression...")
    param_grid = {"model__C": [0.5, 1, 2, 4], "features__tfidf__min_df": [2, 5]}
    with mlflow.start_run(run_name="Logistic Regression (Tuned - Best)"):
        t0 = time.time()
        grid = GridSearchCV(
            build_pipeline(LogisticRegression(max_iter=1000, random_state=42)),
            param_grid,
            cv=5,
            scoring="f1",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        grid_time = time.time() - t0

        best_pipe = grid.best_estimator_
        t0 = time.time()
        pred_test = best_pipe.predict(X_test)
        infer_ms = (time.time() - t0) / len(X_test) * 1000
        pred_train = best_pipe.predict(X_train)

        tr_f1 = float(f1_score(y_train, pred_train))
        te_f1 = float(f1_score(y_test, pred_test))
        te_acc = float(accuracy_score(y_test, pred_test))
        te_prec = float(precision_score(y_test, pred_test))
        te_rec = float(recall_score(y_test, pred_test))
        cv_f1 = float(grid.best_score_)

        mlflow.log_param("model_name", "Logistic Regression (Tuned)")
        mlflow.log_params(grid.best_params_)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("train_f1", round(tr_f1, 4))
        mlflow.log_metric("test_f1", round(te_f1, 4))
        mlflow.log_metric("cv_f1", round(cv_f1, 4))
        mlflow.log_metric("test_accuracy", round(te_acc, 4))
        mlflow.log_metric("train_time_s", round(grid_time, 2))
        mlflow.log_metric("infer_ms_per_review", round(infer_ms, 4))

        # Save serialized best pipeline
        joblib.dump(best_pipe, "models/sentiment_lr.joblib")
        mlflow.log_artifact("models/sentiment_lr.joblib")

        tuned_row = {
            "Model": "Logistic Regression (Tuned ★)",
            "Train F1": round(tr_f1, 4),
            "Test F1": round(te_f1, 4),
            "CV F1 (5-fold)": round(cv_f1, 4),
            "Test Acc": round(te_acc, 4),
            "Test Precision": round(te_prec, 4),
            "Test Recall": round(te_rec, 4),
            "Train time (s)": round(grid_time, 1),
            "ms / review": round(infer_ms, 3),
            "Tuned": "Yes",
        }
        results.append(tuned_row)
        print("Best params:", grid.best_params_, "Best CV F1:", round(cv_f1, 4), "Test F1:", round(te_f1, 4))

    results_df = pd.DataFrame(results)
    results_df.to_csv("reports/classic_models_results.csv", index=False)

    metadata = {
        "model_name": "TF-IDF (1-2 gram) + MetaFeatures + Logistic Regression (Tuned)",
        "version": "1.0.0",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "IMDB Dataset of 50K Movie Reviews",
        "raw_samples": raw_len,
        "deduplicated_samples": len(df),
        "duplicates_removed": dup_count,
        "html_reviews_count": html_count,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "best_params": grid.best_params_,
        "metrics": {
            "train_f1": round(tr_f1, 4),
            "test_f1": round(te_f1, 4),
            "cv_f1_5fold": round(cv_f1, 4),
            "test_accuracy": round(te_acc, 4),
            "test_precision": round(te_prec, 4),
            "test_recall": round(te_rec, 4),
            "inference_ms_per_review": round(infer_ms, 3),
        },
    }
    with open("models/model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n=== Classical Model Comparison ===")
    print(results_df.to_string(index=False))
    return results_df, metadata


if __name__ == "__main__":
    run_training()

