# Phân tích cảm xúc review phim IMDB — Hệ thống Machine Learning ứng dụng End-to-End

Hệ thống Machine Learning và Deep Learning (NLP) hoàn chỉnh, tự động phân loại review phim thành **Tích cực (Positive)**, **Tiêu cực (Negative)**, **Vừa khen vừa chê (Mixed)** hoặc **Không chắc chắn (Uncertain)**. Hệ thống gồm Pipeline Scikit-learn, các đặc trưng tự tạo (Feature Engineering), mô hình **DistilBERT** đã fine-tune, theo dõi thí nghiệm bằng **MLflow**, REST API bằng **FastAPI**, đóng gói bằng **Docker** và **giám sát data drift**.

---

## 1. Tên đề tài & Mô tả

- **Lĩnh vực:** Xử lý ngôn ngữ tự nhiên (NLP) — Phân tích dữ liệu truyền thông & giải trí
- **Bài toán:** Các nền tảng xem phim trực tuyến, rạp chiếu và hãng phim nhận hàng chục nghìn review của khán giả mỗi ngày. Đọc thủ công để nắm phản hồi của khán giả vừa chậm, vừa thiếu nhất quán và không thể mở rộng. Project xây dựng hệ thống tự động phân loại cảm xúc nhị phân và giải thích kết quả theo từng từ, trên bộ dữ liệu **IMDB 50K Movie Reviews**.
- **Vì sao dùng Logistic Regression cho demo và API:** pipeline **TF-IDF (1–2 gram) + MetaFeatures + Logistic Regression** đã tune đạt **91.12% Test F1**, chỉ kém Linear SVM (91.59%) khoảng 0.5 điểm, nhưng có xác suất (`predict_proba`) để tính độ tin cậy, gắn nhãn `uncertain` (độ tin cậy < 0.60) và `mixed` (các vế câu trái chiều); chạy nhanh (`1.023 ms/review`, so với khoảng 83 ms/review của DistilBERT trên CPU); và giải thích được đóng góp của từng từ (`explain()`). DistilBERT mới được fine-tune trên tập con 320 review bằng CPU (không có GPU) nên kết quả chưa so sánh ngang được với các mô hình cổ điển.

---

## 2. Thành viên & Phân công

| Thành viên | Họ tên | MSSV | Phụ trách | Phần thuyết trình |
| :--- | :--- | :--- | :--- | :--- |
| **Thành viên 1 (Nhóm trưởng)** | Võ Phan Bảo Chương | SE196645 | Kiến trúc tổng thể, EDA (`01_eda.ipynb`), `src/features.py` | Bài toán, dữ liệu & tiền xử lý |
| **Thành viên 2** | Phạm Tiến Triển | SE182026 | Mô hình cổ điển & GridSearchCV (`02_classic_models.ipynb`), DistilBERT (`03_distilbert.ipynb`), MLflow | Mô hình & thí nghiệm |
| **Thành viên 3** | Trần Quốc Thái | SE192786 | FastAPI (`api/app.py`), Docker, kiểm thử (`tests/`), giám sát drift (`src/monitor.py`) | Demo, kiểm thử & hạn chế |

### Cách làm việc nhóm

| Công việc | Công cụ |
| :--- | :--- |
| Trao đổi hằng ngày, phân công, gửi file | Nhóm **Zalo** |
| Họp nhóm và thuyết trình | **Google Meet** |
| Quản lý code, chia sẻ dữ liệu và mô hình | **GitHub** (repo ProjectML, mục Releases) |
| Soạn kịch bản thuyết trình | **Google Docs** |
| Ghi lại và so sánh kết quả thí nghiệm | **MLflow** |

**Quy trình:**
- Phân công và báo tiến độ qua nhóm Zalo.
- Trước khi sửa code, mỗi thành viên chạy `git pull` để lấy bản mới nhất; sửa xong thì `git push` và báo lên Zalo.
- Mọi lần train được ghi lại bằng MLflow để cả nhóm cùng xem và so sánh.

---

## 3. Cài đặt

### Yêu cầu
- Python `3.12+` (numpy 2.5 yêu cầu; nhóm chạy trên Python `3.14`)
- Git & VS Code (cài extension Python và Jupyter)

### Các bước cài đặt
```bash
# 1. Clone repository và vào thư mục gốc của project
cd PROJECT

# 2. Tạo và kích hoạt môi trường ảo
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\activate
# Linux / macOS:
# source .venv/bin/activate

# 3. Cài các thư viện (đã cố định phiên bản)
pip install -r requirements.txt
```

---

## 4. Bộ dữ liệu

- **Tên bộ dữ liệu:** IMDB Dataset of 50K Movie Reviews (Andrew Maas và cộng sự, Stanford AI Lab / Kaggle)
- **Nguồn:**
  - Kaggle: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
  - Bản sao trên Hugging Face Hub: https://huggingface.co/datasets/stanfordnlp/imdb
- **Thống kê tóm tắt:**
  - **Kích thước gốc:** `50,000` review phim (`25,000 positive` / `25,000 negative`, cân bằng 50/50).
  - **Các vấn đề chất lượng dữ liệu đã phát hiện & xử lý:**
    1. **Thẻ HTML thừa (`<br />`):** `29,200` review (`58.40%`) chứa thẻ xuống dòng HTML `<br />`, được xóa bằng regex trong `clean_text()`.
    2. **Review trùng lặp:** `418` review trùng (`0.84%`) được loại bỏ **trước khi chia train/test** để tránh rò rỉ dữ liệu, còn lại **`49,582` review duy nhất** (`39,665` Train / `9,917` Test, chia phân tầng 80/20).
    3. **Nhiễu do phủ định & cảm xúc lẫn lộn:** Danh sách stopword chuẩn sẽ xóa `"not"`, `"no"`, `"never"`, làm đảo ngược cảm xúc (`"not good"` $\rightarrow$ `"good"`). Pipeline của nhóm giữ lại toàn bộ từ phủ định và kết hợp với bigram `(1, 2)`.
- **Lệnh tải dữ liệu tự động:**
  ```bash
  python src/data_collection.py
  ```

---

## 5. Kiến trúc Pipeline ML

Được viết theo module trong [`src/features.py`](src/features.py) và [`src/pipeline.py`](src/pipeline.py), dùng `Pipeline` và `ColumnTransformer` của Scikit-learn:

1. **Nhánh văn bản (`TfidfVectorizer`):**
   - `preprocessor=clean_text` tự viết: xóa thẻ HTML `<.*?>`, chuẩn hóa khoảng trắng, chuyển về chữ thường nhưng **giữ lại các từ phủ định**.
   - `ngram_range=(1, 2)`: lấy cả từ đơn và cụm 2 từ như `"not bad"`, `"not good"`, `"waste of"`.
   - `min_df=5` (chọn bằng GridSearchCV), `max_df=0.9`, `sublinear_tf=True` ($1 + \log(\text{tf})$).
2. **Nhánh đặc trưng tự tạo (`MetaFeatures` + `SimpleImputer(strategy="median")` + `StandardScaler`):**
   - Tính trên văn bản gốc, *trước khi* chuyển chữ thường:
     1. `word_count`: tổng số từ trong review.
     2. `exclamation_count`: số dấu `!` (nhiều hơn ở review tiêu cực gay gắt: `1.09` so với `0.88`).
     3. `uppercase_ratio`: tỷ lệ chữ in hoa (bắt được kiểu "la hét" như `"WORST MOVIE EVER"`).
     4. `negation_count`: số từ phủ định (`{"not", "no", "never", "nor", "cannot", "n't"}`; `3.74` ở review tiêu cực so với `2.92` ở review tích cực).
3. **Bộ phân loại:** `LogisticRegression(C=4, max_iter=1000, random_state=42)` đã tune.

---

## 6. Kết quả thí nghiệm (Bảng so sánh tổng hợp)

Tất cả mô hình có giám sát được train và đánh giá trên cùng một **phép chia phân tầng 80/20** (`39,665` train / `9,917` test, `random_state=42`):

| Mô hình | Train F1 | Test F1 | CV F1 (5-Fold) | Test Accuracy | Test Precision | Test Recall | Thời gian train (s) | Dự đoán (ms/review) | Đã tune? | Ghi chú |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression (Baseline)** | 0.9539 | 0.9019 | 0.9038 | 0.9004 | 0.8912 | 0.9130 | 59.1 | 1.049 | Không | Mặc định `C=1.0, min_df=3` |
| **Linear SVM (`LinearSVC`)** | 0.9996 | 0.9159 | 0.9152 | 0.9149 | 0.9090 | 0.9228 | 125.9 | 1.151 | Không | Phân tách tốt trên dữ liệu nhiều chiều, không có sẵn `predict_proba` |
| **Random Forest (`200 cây`)** | 1.0000 | 0.8590 | 0.8442 | 0.8580 | 0.8564 | 0.8616 | 132.9 | 1.030 | Không | Overfitting nặng (chênh lệch F1 `14.1%` trên dữ liệu văn bản thưa) |
| **DistilBERT (`Fine-tune, dropout=0.1`)** | 0.9600 | **0.9194** | 0.9268 (Val) | **0.9167** | 0.9167 | 0.9194 | 214.3 | 83.340 | Có | Chỉ train trên 320 review và test trên 120 review (CPU), chưa so sánh ngang được; chậm hơn ~81 lần trên CPU |
| **Logistic Regression (Đã tune ★)** | **0.9881** | **0.9112** | **0.9130** | **0.9101** | **0.9029** | **0.9196** | **885.4** | **1.023** | **Có** | **Được chọn cho API (`C=4, min_df=5`)** |

---

## 7. Sử dụng API (FastAPI)

### Chạy server trực tiếp
```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```
Mở Swagger UI trên trình duyệt: **http://localhost:8000/docs**

### Danh sách endpoint
| Phương thức | Endpoint | Mô tả |
| :--- | :--- | :--- |
| `GET` | `/health` | Kiểm tra trạng thái dịch vụ và tên mô hình đang dùng |
| `GET` | `/model-info` | Trả về phiên bản mô hình, ngày train, siêu tham số tốt nhất và các chỉ số đánh giá |
| `POST` | `/predict` | Dự đoán cảm xúc: `positive`, `negative`, `mixed` (các vế câu trái chiều, review ≤ 60 từ) hoặc `uncertain` (độ tin cậy < 0.6); ghi lại độ trễ |
| `POST` | `/explain` | Dự đoán cảm xúc và trả về các từ đóng góp nhiều nhất theo hướng tích cực / tiêu cực |
| `POST` | `/batch-predict` | Dự đoán hàng loạt cho một danh sách review |

### Ví dụ lệnh `curl` & kết quả mong đợi

```bash
# 1. Kiểm tra trạng thái
curl -X GET http://localhost:8000/health
# Kết quả mong đợi:
# {"status":"ok","model":"TF-IDF + Logistic Regression"}

# 2. Dự đoán một review
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "An absolute masterpiece, brilliant acting and a beautiful story"}'
# Kết quả mong đợi:
# {"prediction":"positive","confidence":0.977,"latency_ms":2.15}   (latency_ms thay đổi theo máy)

# 3. Review vừa khen vừa chê
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Amazing visuals, awful dialogue."}'
# Kết quả mong đợi:
# {"prediction":"mixed","confidence":0.827,"latency_ms":3.1,
#  "clauses":[{"text":"Amazing visuals","label":"positive","confidence":0.992},
#             {"text":"awful dialogue","label":"negative","confidence":0.993}]}

# 4. Review trung tính
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The movie is two hours long and was released in 2010"}'
# Kết quả mong đợi:
# {"prediction":"uncertain","confidence":0.599,"latency_ms":1.9}
```

### Ví dụ gọi API bằng Python `requests`
```python
import requests

resp = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Worst movie I have ever seen, a total waste of time"}
)
print(resp.json())
# {'prediction': 'negative', 'confidence': 1.0, 'latency_ms': 1.92}
```

---

## 8. Đóng gói bằng Docker

**Yêu cầu:** cài và mở **Docker Desktop**. Nếu đang chạy `uvicorn` hoặc `mlflow ui` trên máy thì tắt trước để không trùng cổng 8000 / 5000.

```bash
# Build và chạy cả API và MLflow UI (lần build đầu khoảng 3 phút)
docker compose up --build
```

| Container | Địa chỉ | Chức năng |
| :--- | :--- | :--- |
| `imdb-sentiment-api` | http://localhost:8000/docs | API dự đoán cảm xúc (FastAPI) |
| `imdb-mlflow-ui` | http://localhost:5000 | Xem kết quả các lần train (MLflow) |

- Nhóm container có tên `project`, image API tên `project-sentiment-api`.
- Image API dùng `python:3.14-slim` và chỉ cài thư viện trong `requirements-api.txt` (không có torch / transformers), nên nhẹ và build nhanh.
- Image chỉ chứa `src/`, `api/` và mô hình `models/sentiment_lr.joblib`; file `.dockerignore` loại bỏ `.venv/`, `data/`, `notebooks/`.
- Log dự đoán được ghi ra thư mục `logs/` trên máy nhờ volume.

```bash
docker ps              # xem các container đang chạy
docker compose down    # tắt và xóa container

# Chạy riêng API, không dùng Compose
docker build -t imdb-sentiment-api .
docker run -p 8000:8000 imdb-sentiment-api
```

---

## 9. Theo dõi thí nghiệm bằng MLflow

Tất cả các lần train (Logistic Regression Baseline, Linear SVM, Random Forest, Logistic Regression tune bằng GridSearchCV, và DistilBERT `dropout=0.1` so với `0.3`) đều được ghi trong `mlruns/`.

```bash
# Mở MLflow UI trên máy (MLflow 3.x, lưu dạng file)
# PowerShell (terminal VS Code trên Windows):
$env:MLFLOW_ALLOW_FILE_STORE="true"
# Nếu dùng CMD thì thay bằng: set MLFLOW_ALLOW_FILE_STORE=true
mlflow ui --backend-store-uri file:./mlruns --port 5000
```
Sau đó mở **http://localhost:5000** để xem siêu tham số, các chỉ số và mô hình đã lưu.

---

## 10. Cấu trúc thư mục

```text
PROJECT/
├── data/
│   ├── raw/IMDB Dataset.csv          # 50.000 review IMDB gốc, chưa chỉnh sửa
│   ├── processed/                    # train.csv & test.csv (80/20) sau khi xóa trùng lặp
│   ├── IMDB Dataset.csv              # File dữ liệu chính mà các notebook dùng
│   ├── train.csv                     # 39.665 mẫu train
│   └── test.csv                      # 9.917 mẫu test
├── notebooks/
│   ├── 01_eda.ipynb                  # Bước 3: Khám phá & kiểm tra dữ liệu
│   ├── 02_classic_models.ipynb       # Bước 6-8, 10-13: LR, SVM, RF, GridSearch, phân tích lỗi & demo
│   └── 03_distilbert.ipynb           # Bước 9: Fine-tune DistilBERT, loss curve & regularization
├── src/
│   ├── __init__.py
│   ├── data_collection.py            # Script tải dữ liệu (tái lập được)
│   ├── features.py                   # Bước 4: clean_text() & transformer MetaFeatures
│   ├── pipeline.py                   # Bước 5: build_pipeline() & explain()
│   ├── train.py                      # Script train toàn bộ & ghi log MLflow
│   ├── predict.py                    # Hàm dự đoán đơn lẻ & hàng loạt
│   ├── monitor.py                    # Giám sát data drift (kiểm định KS 2 mẫu) & log dự đoán
│   └── generate_figures_and_bert.py  # Vẽ toàn bộ hình cho báo cáo & fine-tune DistilBERT
├── api/
│   ├── __init__.py
│   └── app.py                        # Bước 13: Ứng dụng FastAPI (/health, /predict, /explain, /model-info, /batch-predict)
├── models/
│   ├── sentiment_lr.joblib           # Pipeline TF-IDF + MetaFeatures + Logistic Regression đã tune
│   ├── model_metadata.json           # Phiên bản mô hình, chỉ số & siêu tham số tốt nhất
│   └── distilbert/                   # Checkpoint DistilBERT đã fine-tune & tokenizer
├── tests/
│   └── test_pipeline.py              # 9 unit test pytest cho features, pipeline, nhãn mixed, drift & API
├── reports/
│   ├── Project_Proposal.md           # Đề xuất project (Giai đoạn 1)
│   ├── Technical_Report.md           # Báo cáo kỹ thuật đầy đủ (Giai đoạn 4)
│   ├── master_comparison_table.csv   # Bảng chỉ số đầy đủ của tất cả mô hình
│   ├── monitoring_report.json        # Báo cáo data drift (KS-test)
│   └── *.png                         # Hình EDA, Confusion Matrix, TF-IDF & Loss Curve
├── mlruns/                           # Nơi MLflow lưu các thí nghiệm
├── Dockerfile                        # Định nghĩa container cho dịch vụ FastAPI (Python 3.14)
├── .dockerignore                     # Loại .venv, data, notebooks khỏi lúc build image
├── docker-compose.yml                # Chạy nhiều dịch vụ (API + MLflow UI)
├── pytest.ini                        # Cấu hình pytest
├── requirements.txt                  # Thư viện Python đầy đủ (đã cố định phiên bản)
├── requirements-api.txt              # Thư viện tối thiểu cho API trong Docker
└── README.md                         # Tài liệu project
```

---

## 11. Tái lập kết quả

Để chạy lại toàn bộ quy trình, từ lấy dữ liệu gốc đến train, đánh giá và unit test, với seed cố định (`random_state=42`):

```bash
# 1. Train các mô hình cổ điển + GridSearchCV 5-fold + ghi log MLflow
python src/train.py

# 2. Vẽ toàn bộ hình EDA/đánh giá và fine-tune DistilBERT
python src/generate_figures_and_bert.py

# 3. Kiểm tra data drift
python src/monitor.py

# 4. Chạy bộ unit test (9 test)
pytest tests/ -v
```
