# numpy 2.5.x cần Python >= 3.12; dùng cùng Python 3.14 như máy phát triển
FROM python:3.14-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=models/sentiment_lr.joblib \
    METADATA_PATH=models/model_metadata.json \
    PREDICTION_LOG_PATH=logs/predictions.csv

# Chỉ cài thư viện cần cho API (nhẹ, build nhanh)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/
COPY api/ ./api/
COPY models/sentiment_lr.joblib ./models/sentiment_lr.joblib
COPY models/model_metadata.json ./models/model_metadata.json

RUN mkdir -p logs

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

