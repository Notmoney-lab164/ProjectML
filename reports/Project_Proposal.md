# Phase 1: Project Proposal — IMDB Movie Review Sentiment Analysis

**Course:** MDS301 — Machine Learning in Data Sciences  
**Project Title:** Automated Sentiment Analysis & Word-Level Attribution on IMDB Movie Reviews  

---

## 1. Problem Statement
Streaming platforms, cinema chains, and film production studios collect tens of thousands of user-written movie reviews every week. Reading and categorizing these reviews manually is slow, expensive, and prone to human fatigue. Our system automatically classifies English movie reviews into **Positive** or **Negative** sentiment (while flagging **Mixed / Uncertain** reviews via confidence thresholding) and highlights the exact words/phrases driving each prediction.

## 2. Machine Learning Formulation
- **Primary Problem Type:** Supervised Binary Text Classification (Natural Language Processing).
- **Justification:** Each review $x_i$ is paired with a ground-truth discrete binary sentiment label $y_i \in \{0, 1\}$ derived from viewer ratings ($\le 4/10 \rightarrow \text{negative}$, $\ge 7/10 \rightarrow \text{positive}$).
- **Model Families Investigated:**
  1. **TF-IDF (1–2 grams) + Custom MetaFeatures + Logistic Regression** (Linear probabilistic baseline & tuned production model).
  2. **TF-IDF + MetaFeatures + Linear SVM (`LinearSVC`)** (Maximum-margin linear classifier in high-dimensional sparse space).
  3. **TF-IDF + MetaFeatures + Random Forest** (Tree-based ensemble to study bias–variance and overfitting on sparse text).
  4. **Fine-tuned `DistilBERT` (`distilbert-base-uncased`)** (Deep learning Transformer encoder capturing bidirectional context).

## 3. Dataset Description
- **Name:** IMDB Dataset of 50K Movie Reviews
- **Source:** [Kaggle IMDB 50K Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) / [Stanford AI Lab (`stanfordnlp/imdb`)](https://huggingface.co/datasets/stanfordnlp/imdb)
- **Scale:** `50,000` samples (`25,000` positive, `25,000` negative).
- **Data Quality Challenges:**
  - `29,200` reviews (`58.40%`) contain raw HTML `<br />` tags.
  - `418` duplicate reviews (`0.84%`) must be removed before splitting to avoid train/test data leakage.
  - Negation expressions (`"not bad"`, `"never boring"`), mixed reviews (`"great acting but terrible plot"`), and sarcasm.

## 4. Quantitative Success Criteria
1. **Predictive Performance:** Achieve **Test F1-Score $\ge 0.90$** and **Test Accuracy $\ge 0.90$** on the held-out 20% test set (`9,917` unique reviews).
2. **Real-Time Inference Latency:** Achieve CPU inference latency **$\le 10\text{ ms}$ per review** for the deployed REST API.

## 5. Project Plan & Task Division

| Week | Phase | Tasks | Responsible Member |
| :---: | :--- | :--- | :--- |
| **Week 1–2** | Phase 1: Scoping & EDA | Dataset acquisition, Data Audit (`01_eda.ipynb`), `src/features.py` | Member A (Võ Phan Bảo Chương) |
| **Week 3–4** | Phase 2a: Classical Pipeline | `src/pipeline.py`, LR, LinearSVC, Random Forest, 5-Fold GridSearchCV (`02_classic_models.ipynb`) | Member B |
| **Week 5** | Phase 2b: Deep Learning | Fine-tune `DistilBERT`, loss curves, Dropout (`0.1` vs `0.3`) & Early Stopping (`03_distilbert.ipynb`) | Member C |
| **Week 6–7** | Phase 3–4: MLOps & Report | FastAPI (`api/app.py`), Docker, MLflow tracking, Drift Monitoring (`src/monitor.py`), Final Report & Demo | Member D & All |

