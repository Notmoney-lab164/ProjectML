# IMDB Movie Review Sentiment Analysis — End-to-End Applied ML System

An end-to-end Production Machine Learning & Deep Learning NLP system that automatically classifies movie reviews as **Positive**, **Negative**, or **Mixed / Uncertain**, complete with a Scikit-learn Pipeline, custom Feature Engineering, Fine-tuned **DistilBERT**, **MLflow** Experiment Tracking, **FastAPI** REST Service, **Docker** Containerization, and **Data Drift Monitoring**.

---

## 1. Project Title & Description

- **Domain:** Natural Language Processing (NLP) — Media & Entertainment Analytics
- **Problem Statement:** Streaming platforms, cinemas, and film studios receive tens of thousands of viewer reviews daily. Manually reading reviews to gauge audience reception is slow, inconsistent, and unscalable. This project builds an automated binary sentiment classification and word-level explanation system on the **IMDB 50K Movie Reviews** dataset.
- **Why Logistic Regression for Live Demo & Production API:** While fine-tuned **DistilBERT** achieves the highest accuracy on full GPU training (~92.8% F1), our tuned **TF-IDF (1–2 gram) + MetaFeatures + Logistic Regression** pipeline reaches **91.12% Test F1** (within ~1.7% of DistilBERT) while running **~80× faster on CPU (`1.023 ms/review`)**, providing calibrated probabilities (`predict_proba`) for uncertainty detection (`confidence < 0.60`), and offering transparent word-level attribution (`explain()`).

---

## 2. Team Members & Task Division

| Member | Student Name | Roll Number | Responsibility | Presentation Section |
| :--- | :--- | :--- | :--- | :--- |
| **Member 1 (Lead)** | Võ Phan Bảo Chương | SE196645 | End-to-End Architecture, EDA (`01_eda.ipynb`), `src/features.py`, MLOps | Problem Statement, Data Audit & Feature Engineering |
| **Member 2** | Member B | SEXXXXXX | `src/pipeline.py`, Classical Models & 5-Fold GridSearchCV (`02_classic_models.ipynb`) | Classical Models & Bias–Variance Analysis |
| **Member 3** | Member C | SEXXXXXX | Deep Learning `DistilBERT` Fine-Tuning & Regularization (`03_distilbert.ipynb`) | Deep Learning Architecture, Loss Curves & Regularization |
| **Member 4** | Member D | SEXXXXXX | Error Analysis, FastAPI (`api/app.py`), Docker & Live Demo | Master Comparison Table, Live Demo & Limitations |

---

## 3. Installation

### Prerequisites
- Python `3.10+` (tested on Python `3.11` and `3.13`)
- Git & VS Code (with Python and Jupyter extensions)

### Step-by-Step Setup
```bash
# 1. Clone the repository and navigate to project root
cd PROJECT

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\activate
# Linux / macOS:
# source .venv/bin/activate

# 3. Install pinned dependencies
pip install -r requirements.txt
```

---

## 4. Dataset

- **Dataset Name:** IMDB Dataset of 50K Movie Reviews (Andrew Maas et al., Stanford AI Lab / Kaggle)
- **Source URLs:**
  - Kaggle: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
  - Hugging Face Hub Mirror: https://huggingface.co/datasets/stanfordnlp/imdb
- **Summary Statistics:**
  - **Raw Size:** `50,000` movie reviews (`25,000 positive` / `25,000 negative` — 50/50 balanced).
  - **Data Quality Challenges Identified & Resolved:**
    1. **HTML Artifacts (`<br />`):** `29,200` reviews (`58.40%`) contain raw HTML line-break tags `<br />`, stripped via regex in `clean_text()`.
    2. **Duplicate Reviews:** `418` duplicate reviews (`0.84%`) removed **prior to train/test splitting** to prevent data leakage, leaving **`49,582` unique reviews** (`39,665` Train / `9,917` Test at 80/20 stratified split).
    3. **Negation & Mixed Sentiment Noise:** Standard stopword lists remove `"not"`, `"no"`, `"never"`, reversing sentiment polarity (`"not good"` $\rightarrow$ `"good"`). Our pipeline preserves all negation words and pairs them with bigrams `(1, 2)`.
- **Automatic Download Command:**
  ```bash
  python src/data_collection.py
  ```

---

## 5. ML Pipeline Architecture

Implemented modularly in [`src/features.py`](src/features.py) and [`src/pipeline.py`](src/pipeline.py) using Scikit-learn's `Pipeline` and `ColumnTransformer`:

1. **Text Branch (`TfidfVectorizer`):**
   - Custom `preprocessor=clean_text`: strips HTML tags `<.*?>`, normalizes whitespace, converts to lowercase while **preserving negation tokens**.
   - `ngram_range=(1, 2)`: captures both unigrams and bigrams such as `"not bad"`, `"not good"`, `"waste of"`.
   - `min_df=5` (tuned via GridSearchCV), `max_df=0.9`, `sublinear_tf=True` ($1 + \log(\text{tf})$).
2. **Engineered Meta-Features Branch (`MetaFeatures` + `SimpleImputer(strategy="median")` + `StandardScaler`):**
   - Computed on raw text *before* lowercasing:
     1. `word_count`: Total number of words in the review.
     2. `exclamation_count`: Count of `!` marks (higher in intense negative reviews: `1.09` vs `0.88`).
     3. `uppercase_ratio`: Fraction of uppercase characters (captures shouting like `"WORST MOVIE EVER"`).
     4. `negation_count`: Count of negation tokens (`{"not", "no", "never", "nor", "cannot", "n't"}`, `3.74` in negative vs `2.92` in positive reviews).
3. **Classifier Head:** Tuned `LogisticRegression(C=4, max_iter=1000, random_state=42)`.

---

## 6. Experiment Results (Master Comparison Table)

All supervised models were trained and evaluated on the exact same **80/20 stratified split** (`39,665` train / `9,917` test, `random_state=42`):

| Model | Train F1 | Test F1 | CV F1 (5-Fold) | Test Accuracy | Test Precision | Test Recall | Train Time (s) | Inference (ms/review) | Tuned? | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression (Baseline)** | 0.9539 | 0.9019 | 0.9038 | 0.9004 | 0.8912 | 0.9130 | 59.1 | 1.049 | No | Default `C=1.0, min_df=3` |
| **Linear SVM (`LinearSVC`)** | 0.9996 | 0.9159 | 0.9152 | 0.9149 | 0.9090 | 0.9228 | 125.9 | 1.151 | No | Strong high-dim margin, no native `predict_proba` |
| **Random Forest (`200 trees`)** | 1.0000 | 0.8590 | 0.8442 | 0.8580 | 0.8564 | 0.8616 | 132.9 | 1.030 | No | Severe overfitting (`14.1%` F1 gap on sparse text) |
| **DistilBERT (`Fine-tuned, dropout=0.1`)** | 0.9600 | **0.9194** | 0.9268 (Val) | **0.9167** | 0.9167 | 0.9194 | 214.3 | 83.340 | Yes | Highest accuracy (`models/distilbert/`), ~81× slower on CPU |
| **Logistic Regression (Tuned ★)** | **0.9881** | **0.9112** | **0.9130** | **0.9101** | **0.9029** | **0.9196** | **885.4** | **1.023** | **Yes** | **Selected for Production API (`C=4, min_df=5`)** |

---

## 7. API Usage (FastAPI)

### Start the Server Natively
```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```
Open Swagger UI in your browser: **http://localhost:8000/docs**

### Endpoints Overview
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Health check returning service status and active model name |
| `GET` | `/model-info` | Returns model version, training date, best hyperparameters, and evaluation metrics |
| `POST` | `/predict` | Predicts sentiment (`positive`, `negative`, or `mixed / uncertain` if confidence < 0.6) and logs latency |
| `POST` | `/explain` | Predicts sentiment and returns top positive/negative contributing words |
| `POST` | `/batch-predict` | Batch inference for a list of movie reviews |

### Example `curl` Requests & Expected Responses

```bash
# 1. Health Check
curl -X GET http://localhost:8000/health
# Expected Response:
# {"status":"ok","model":"TF-IDF + Logistic Regression"}

# 2. Single Review Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "An absolute masterpiece, brilliant acting and a beautiful story"}'
# Expected Response:
# {"prediction":"positive","confidence":0.996,"latency_ms":2.15}

# 3. Negation Handling Test
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Honestly, it was not bad at all"}'
# Expected Response:
# {"prediction":"positive","confidence":0.829,"latency_ms":1.84}
```

### Example Python `requests` Snippet
```python
import requests

resp = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Worst movie I have ever seen, a total waste of time"}
)
print(resp.json())
# {'prediction': 'negative', 'confidence': 0.999, 'latency_ms': 1.92}
```

---

## 8. Docker Containerization

```bash
# Build the Docker image
docker build -t imdb-sentiment-api .

# Run the container on port 8000
docker run -p 8000:8000 imdb-sentiment-api

# Or launch both the API and MLflow UI via Docker Compose
docker compose up --build
```

---

## 9. MLflow Experiment Tracking

All training runs (Logistic Regression Baseline, Linear SVM, Random Forest, GridSearchCV Tuned Logistic Regression, and DistilBERT `dropout=0.1` vs `0.3`) are logged in `mlruns/`.

```bash
# Launch MLflow UI locally (for MLflow 3.x with file store)
set MLFLOW_ALLOW_FILE_STORE=true
mlflow ui --backend-store-uri file:./mlruns --port 5000
```
Then visit **http://localhost:5000** to inspect hyperparameters, metrics, and logged model artifacts.

---

## 10. Repository Structure

```text
PROJECT/
├── data/
│   ├── raw/IMDB Dataset.csv          # Original unmodified 50,000 IMDB reviews
│   ├── processed/                    # Deduplicated 80/20 train.csv & test.csv splits
│   ├── IMDB Dataset.csv              # Main dataset path used by notebooks
│   ├── train.csv                     # 39,665 training samples
│   └── test.csv                      # 9,917 test samples
├── notebooks/
│   ├── 01_eda.ipynb                  # Step 3: Exploratory Data Analysis & Data Audit
│   ├── 02_classic_models.ipynb       # Steps 6-8, 10-13: LR, SVM, RF, GridSearch, Error Analysis & Demo
│   └── 03_distilbert.ipynb           # Step 9: DistilBERT fine-tuning, loss curves & regularization
├── src/
│   ├── __init__.py
│   ├── data_collection.py            # Reproducible dataset download script
│   ├── features.py                   # Step 4: clean_text() & MetaFeatures transformer
│   ├── pipeline.py                   # Step 5: build_pipeline() & explain()
│   ├── train.py                      # End-to-end training & MLflow logging script
│   ├── predict.py                    # Single & batch inference utilities
│   ├── monitor.py                    # Two-sample KS data drift & prediction log monitoring
│   └── generate_figures_and_bert.py  # Generates all report figures & fine-tunes DistilBERT
├── api/
│   ├── __init__.py
│   └── app.py                        # Step 13: FastAPI application (/health, /predict, /explain, /model-info, /batch-predict)
├── models/
│   ├── sentiment_lr.joblib           # Serialized tuned TF-IDF + MetaFeatures + Logistic Regression pipeline
│   ├── model_metadata.json           # Model version, metrics & best hyperparameters
│   └── distilbert/                   # Fine-tuned DistilBERT checkpoint & tokenizer
├── tests/
│   └── test_pipeline.py              # 8 pytest unit tests covering features, pipeline, drift & API
├── reports/
│   ├── Project_Proposal.md           # Phase 1 Project Proposal document
│   ├── Technical_Report.md           # Phase 4 Comprehensive Technical Report
│   ├── master_comparison_table.csv   # Full metrics table across all models
│   ├── monitoring_report.json        # KS-test data drift report
│   └── *.png                         # Generated EDA, Confusion Matrix, TF-IDF & Loss Curve figures
├── mlruns/                           # MLflow experiment tracking store
├── Dockerfile                        # Container definition for FastAPI service
├── docker-compose.yml                # Multi-service orchestration (API + MLflow UI)
├── pytest.ini                        # Pytest configuration
├── requirements.txt                  # Pinned Python dependencies
└── README.md                         # Project documentation
```

---

## 11. Reproducibility

To reproduce the entire pipeline from raw data acquisition through model training, evaluation, and unit testing with fixed random seed (`random_state=42`):

```bash
# 1. Run end-to-end classical training + 5-fold GridSearchCV + MLflow logging
python src/train.py

# 2. Generate all EDA/Evaluation figures and fine-tune DistilBERT
python src/generate_figures_and_bert.py

# 3. Run data drift monitoring check
python src/monitor.py

# 4. Run unit test suite (8 tests)
pytest tests/ -v
```

