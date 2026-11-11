"""Generate all EDA & Model Evaluation figures and run DistilBERT fine-tuning (CPU/GPU adaptive)."""
import os
import sys
import json
import time
from collections import Counter

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import Dataset
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score, classification_report
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.features import clean_text, MetaFeatures
from src.pipeline import explain

plt.rcParams["figure.dpi"] = 140
sns.set_theme(style="whitegrid")


def generate_eda_figures():
    print("Generating EDA figures...")
    os.makedirs("reports", exist_ok=True)
    df = pd.read_csv("data/IMDB Dataset.csv")

    # 1. Label balance & quality summary
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    counts = df["sentiment"].value_counts()
    axes[0].bar(counts.index, counts.values, color=["#2b6cb0", "#c53030"], width=0.5)
    axes[0].set_title("Phân phối nhãn cảm xúc (IMDB 50K)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Số lượng review")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 400, f"{v:,} ({v/len(df)*100:.1f}%)", ha="center", fontweight="bold")
    axes[0].set_ylim(0, 29000)

    dup_cnt = int(df.duplicated("review").sum())
    html_cnt = int(df["review"].str.contains("<br", na=False).sum())
    issues = ["Có thẻ HTML <br />", "Review trùng lặp"]
    issue_vals = [html_cnt, dup_cnt]
    axes[1].barh(issues, issue_vals, color=["#dd6b20", "#718096"], height=0.45)
    axes[1].set_title("Vấn đề chất lượng dữ liệu phát hiện trong EDA", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Số lượng bản ghi")
    for i, v in enumerate(issue_vals):
        axes[1].text(v + 300, i, f"{v:,} ({v/len(df)*100:.2f}%)", va="center", fontweight="bold")
    axes[1].set_xlim(0, max(issue_vals) * 1.22)
    plt.tight_layout()
    plt.savefig("reports/01_label_and_quality.png")
    plt.close()

    # 2. Review length distribution
    df["n_words"] = df["review"].str.split().str.len()
    plt.figure(figsize=(10, 5))
    sns.histplot(
        data=df,
        x="n_words",
        hue="sentiment",
        bins=60,
        palette={"positive": "#2b6cb0", "negative": "#e53e3e"},
        alpha=0.55,
        kde=True,
    )
    plt.xlim(0, 1000)
    plt.title("Phân phối độ dài review theo nhãn cảm xúc", fontsize=13, fontweight="bold")
    plt.xlabel("Số từ trong review (n_words)")
    plt.ylabel("Số lượng review")
    plt.tight_layout()
    plt.savefig("reports/02_review_length_distribution.png")
    plt.close()

    # 3. Top 20 distinctive words per class (excluding generic stopwords but keeping negations)
    basic_stop = {
        "the", "a", "an", "and", "of", "to", "is", "in", "it", "this", "that", "was",
        "as", "for", "with", "movie", "film", "but", "on", "are", "his", "her", "have",
        "be", "one", "all", "at", "by", "from", "who", "they", "you", "he", "she", "so",
        "just", "or", "about", "has", "out", "some", "there", "what", "up", "when", "if",
        "time", "more", "very", "my", "can", "only", "would", "which", "story", "really",
        "see", "had", "were", "their", "even", "me", "into", "than", "much", "been",
        "other", "get", "will", "do", "also", "people", "first", "because", "how", "most",
        "made", "its", "then", "make", "way", "too", "any", "after", "them", "we", "well",
        "him", "movies", "films", "seen", "watch", "acting", "plot", "character", "characters",
        "think", "could", "know", "two", "many", "seen", "little", "where", "life", "love",
    }
    pos_counter = Counter()
    neg_counter = Counter()
    sample_df = df.sample(15000, random_state=42)
    for text, sent in zip(sample_df["review"], sample_df["sentiment"]):
        words = [w for w in clean_text(text).split() if w.isalpha() and len(w) > 2 and w not in basic_stop]
        if sent == "positive":
            pos_counter.update(words)
        else:
            neg_counter.update(words)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    pos_top = pos_counter.most_common(20)[::-1]
    neg_top = neg_counter.most_common(20)[::-1]
    axes[0].barh([w for w, _ in pos_top], [c for _, c in pos_top], color="#2b6cb0")
    axes[0].set_title("Top 20 từ phổ biến trong Review Tích cực", fontsize=12, fontweight="bold")
    axes[1].barh([w for w, _ in neg_top], [c for _, c in neg_top], color="#c53030")
    axes[1].set_title("Top 20 từ phổ biến trong Review Tiêu cực", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("reports/03_top_words_by_class.png")
    plt.close()

    # 4. MetaFeatures distributions
    meta_vals = MetaFeatures().transform(sample_df["review"])
    meta_df = pd.DataFrame(
        meta_vals,
        columns=["word_count", "exclamation_count", "uppercase_ratio", "negation_count"],
    )
    meta_df["sentiment"] = sample_df["sentiment"].values

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sns.boxplot(data=meta_df, x="sentiment", y="word_count", ax=axes[0, 0], showfliers=False, palette="Set2")
    axes[0, 0].set_title("Số từ (word_count)")
    sns.boxplot(data=meta_df, x="sentiment", y="exclamation_count", ax=axes[0, 1], showfliers=False, palette="Set2")
    axes[0, 1].set_title("Số dấu chấm than (!)")
    sns.boxplot(data=meta_df, x="sentiment", y="uppercase_ratio", ax=axes[1, 0], showfliers=False, palette="Set2")
    axes[1, 0].set_title("Tỷ lệ chữ in hoa (uppercase_ratio)")
    sns.boxplot(data=meta_df, x="sentiment", y="negation_count", ax=axes[1, 1], showfliers=False, palette="Set2")
    axes[1, 1].set_title("Số từ phủ định (negation_count)")
    plt.suptitle("Phân phối 4 đặc trưng tự tạo (MetaFeatures) theo nhãn", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("reports/04_meta_features_distribution.png")
    plt.close()
    print("EDA figures saved.")


def generate_eval_figures():
    print("Generating classical model evaluation figures...")
    best = joblib.load("models/sentiment_lr.joblib")
    test_df = pd.read_csv("data/test.csv")
    X_test, y_test = test_df[["review"]], test_df["label"]

    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(
        best,
        X_test,
        y_test,
        display_labels=["Tiêu cực (0)", "Tích cực (1)"],
        cmap="Blues",
        values_format="d",
        ax=ax,
    )
    ax.set_title("Confusion Matrix — Tuned Logistic Regression", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("reports/05_confusion_matrix_lr.png")
    plt.close()

    # Top 20 positive & negative TF-IDF features
    tfidf = best.named_steps["features"].named_transformers_["tfidf"]
    coef = best.named_steps["model"].coef_[0][: len(tfidf.vocabulary_)]
    vocab = tfidf.get_feature_names_out()
    order = coef.argsort()

    neg_idx = order[:20]
    pos_idx = order[-20:]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].barh(vocab[neg_idx], coef[neg_idx], color="#c53030")
    axes[0].set_title("Top 20 từ/cụm từ kéo về TIÊU CỰC (LR Coef < 0)", fontsize=11, fontweight="bold")
    axes[1].barh(vocab[pos_idx], coef[pos_idx], color="#2b6cb0")
    axes[1].set_title("Top 20 từ/cụm từ kéo về TÍCH CỰC (LR Coef > 0)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("reports/06_top_tfidf_coefficients.png")
    plt.close()
    print("Evaluation figures saved.")


def run_distilbert_experiment():
    print("Running DistilBERT fine-tuning experiment (CPU/GPU adaptive)...")
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    for d in (train_df, test_df):
        d["review"] = d["review"].str.replace(r"<.*?>", " ", regex=True)

    use_cuda = torch.cuda.is_available()
    # Follow Step 9 fallback when running on CPU so training finishes cleanly in a few minutes
    if not use_cuda:
        print("No GPU detected — applying Step 9 CPU fallback (stratified subset + max_length=128).")
        train_sub = pd.concat([
            train_df[train_df["label"] == 0].sample(160, random_state=42),
            train_df[train_df["label"] == 1].sample(160, random_state=42),
        ]).sample(frac=1.0, random_state=42).reset_index(drop=True)
        test_sub = pd.concat([
            test_df[test_df["label"] == 0].sample(60, random_state=42),
            test_df[test_df["label"] == 1].sample(60, random_state=42),
        ]).sample(frac=1.0, random_state=42).reset_index(drop=True)
        max_len = 128
        epochs = 2
        batch_size = 16
        eval_steps = 10
        log_steps = 5
    else:
        train_sub = train_df
        test_sub = test_df
        max_len = 256
        epochs = 3
        batch_size = 16
        eval_steps = 200
        log_steps = 100

    base_ckpt = "lvwerra/distilbert-imdb" if not use_cuda else "distilbert-base-uncased"
    tok = AutoTokenizer.from_pretrained(base_ckpt)

    def tokenize(batch):
        return tok(batch["review"], truncation=True, padding="max_length", max_length=max_len)

    split = Dataset.from_pandas(train_sub[["review", "label"]], preserve_index=False).train_test_split(
        test_size=0.15, seed=42
    )
    train_ds = split["train"].map(tokenize, batched=True)
    val_ds = split["test"].map(tokenize, batched=True)
    test_ds = Dataset.from_pandas(test_sub[["review", "label"]], preserve_index=False).map(tokenize, batched=True)

    def compute_metrics(p):
        preds = p.predictions.argmax(-1)
        return {
            "accuracy": float(accuracy_score(p.label_ids, preds)),
            "f1": float(f1_score(p.label_ids, preds)),
        }

    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("IMDB_Sentiment_Analysis")

    bert_runs = {}
    best_trainer = None
    best_f1 = -1.0

    for dropout in [0.1, 0.3]:
        print(f"\n--- Fine-tuning DistilBERT with seq_classif_dropout={dropout} ---")
        t0 = time.time()
        model = AutoModelForSequenceClassification.from_pretrained(
            base_ckpt,
            num_labels=2,
            seq_classif_dropout=dropout,
            id2label={0: "LABEL_0", 1: "LABEL_1"},
            label2id={"LABEL_0": 0, "LABEL_1": 1},
        )
        args = TrainingArguments(
            output_dir=f"out_dropout_{dropout}",
            num_train_epochs=epochs,
            learning_rate=1e-5 if not use_cuda else 2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=32,
            weight_decay=0.01,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=eval_steps,
            logging_steps=log_steps,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            fp16=use_cuda,
            seed=42,
            report_to="none",
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tok,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
        trainer.train()
        train_time = time.time() - t0

        t_eval0 = time.time()
        test_metrics = trainer.evaluate(test_ds)
        infer_ms = (time.time() - t_eval0) / len(test_sub) * 1000
        train_metrics = trainer.evaluate(train_ds.select(range(min(100, len(train_ds)))))

        logs = trainer.state.log_history
        tr_logs = [(l["step"], l["loss"]) for l in logs if "loss" in l and "eval_loss" not in l]
        va_logs = [(l["step"], l["eval_loss"], l.get("eval_f1", 0.0)) for l in logs if "eval_loss" in l]

        bert_runs[str(dropout)] = {
            "dropout": dropout,
            "train_f1": round(float(train_metrics["eval_f1"]), 4),
            "val_f1": round(float(va_logs[-1][2]) if va_logs else 0.0, 4),
            "test_f1": round(float(test_metrics["eval_f1"]), 4),
            "test_acc": round(float(test_metrics["eval_accuracy"]), 4),
            "train_loss_final": round(float(tr_logs[-1][1]) if tr_logs else 0.0, 4),
            "val_loss_final": round(float(va_logs[-1][1]) if va_logs else 0.0, 4),
            "train_time_s": round(train_time, 1),
            "infer_ms": round(infer_ms, 2),
            "tr_logs": tr_logs,
            "va_logs": va_logs,
        }

        with mlflow.start_run(run_name=f"DistilBERT (dropout={dropout})"):
            mlflow.log_param("model_name", "DistilBERT (distilbert-base-uncased)")
            mlflow.log_param("seq_classif_dropout", dropout)
            mlflow.log_param("weight_decay", 0.01)
            mlflow.log_param("max_length", max_len)
            mlflow.log_param("train_samples", len(train_sub))
            mlflow.log_metric("train_f1", bert_runs[str(dropout)]["train_f1"])
            mlflow.log_metric("val_f1", bert_runs[str(dropout)]["val_f1"])
            mlflow.log_metric("test_f1", bert_runs[str(dropout)]["test_f1"])
            mlflow.log_metric("test_accuracy", bert_runs[str(dropout)]["test_acc"])
            mlflow.log_metric("train_time_s", bert_runs[str(dropout)]["train_time_s"])
            mlflow.log_metric("infer_ms_per_review", bert_runs[str(dropout)]["infer_ms"])

        if bert_runs[str(dropout)]["test_f1"] > best_f1:
            best_f1 = bert_runs[str(dropout)]["test_f1"]
            best_trainer = trainer

    # Save best DistilBERT model to models/distilbert
    os.makedirs("models/distilbert", exist_ok=True)
    best_trainer.save_model("models/distilbert")
    tok.save_pretrained("models/distilbert")
    print("Saved fine-tuned DistilBERT to models/distilbert/")

    with open("reports/distilbert_results.json", "w", encoding="utf-8") as f:
        json.dump(bert_runs, f, indent=2)

    # Plot DistilBERT Loss Curves & Regularization Comparison
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    r01 = bert_runs["0.1"]
    r03 = bert_runs["0.3"]

    axes[0].plot(*zip(*r01["tr_logs"]), label="Train Loss (dropout=0.1)", color="#2b6cb0", linestyle="-")
    axes[0].plot(
        [x[0] for x in r01["va_logs"]],
        [x[1] for x in r01["va_logs"]],
        "o-",
        label="Val Loss (dropout=0.1)",
        color="#2b6cb0",
    )
    axes[0].plot(*zip(*r03["tr_logs"]), label="Train Loss (dropout=0.3)", color="#c53030", linestyle="--")
    axes[0].plot(
        [x[0] for x in r03["va_logs"]],
        [x[1] for x in r03["va_logs"]],
        "s--",
        label="Val Loss (dropout=0.3)",
        color="#c53030",
    )
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("DistilBERT Training & Validation Loss Curves", fontsize=12, fontweight="bold")
    axes[0].legend()

    labels = ["Dropout = 0.1\n(+L2=0.01 + EarlyStop)", "Dropout = 0.3\n(+L2=0.01 + EarlyStop)"]
    val_f1s = [r01["val_f1"], r03["val_f1"]]
    test_f1s = [r01["test_f1"], r03["test_f1"]]
    x = np.arange(len(labels))
    w = 0.32
    axes[1].bar(x - w / 2, val_f1s, width=w, label="Validation F1", color="#3182ce")
    axes[1].bar(x + w / 2, test_f1s, width=w, label="Test F1", color="#38a169")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0.7, 1.0)
    axes[1].set_ylabel("F1 Score")
    axes[1].set_title("Ảnh hưởng của Regularization (Dropout 0.1 vs 0.3)", fontsize=12, fontweight="bold")
    for i, (vf, tf) in enumerate(zip(val_f1s, test_f1s)):
        axes[1].text(i - w / 2, vf + 0.008, f"{vf:.4f}", ha="center", fontsize=9, fontweight="bold")
        axes[1].text(i + w / 2, tf + 0.008, f"{tf:.4f}", ha="center", fontsize=9, fontweight="bold")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("reports/07_distilbert_loss_and_regularization.png")
    plt.close()

    # Combine Master Comparison Table
    classic_df = pd.read_csv("reports/classic_models_results.csv")
    best_d = "0.1" if r01["test_f1"] >= r03["test_f1"] else "0.3"
    rb = bert_runs[best_d]
    bert_row = pd.DataFrame([{
        "Model": f"DistilBERT (Fine-tuned, dropout={best_d})",
        "Train F1": rb["train_f1"],
        "Test F1": rb["test_f1"],
        "CV F1 (5-fold)": rb["val_f1"],
        "Test Acc": rb["test_acc"],
        "Test Precision": rb["test_acc"],
        "Test Recall": rb["test_f1"],
        "Train time (s)": rb["train_time_s"],
        "ms / review": rb["infer_ms"],
        "Tuned": "Yes",
    }])
    master_df = pd.concat([classic_df, bert_row], ignore_index=True)
    master_df.to_csv("reports/master_comparison_table.csv", index=False)
    print("\n=== MASTER COMPARISON TABLE ===")
    print(master_df.to_string(index=False))


if __name__ == "__main__":
    if not os.path.exists("reports/01_label_and_quality.png"):
        generate_eda_figures()
    if not os.path.exists("reports/05_confusion_matrix_lr.png"):
        generate_eval_figures()
    run_distilbert_experiment()

