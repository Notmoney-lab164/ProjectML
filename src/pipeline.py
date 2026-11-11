import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from src.features import clean_text, MetaFeatures


def build_pipeline(model, max_features=None, min_df=3, max_df=0.9, ngram_range=(1, 2)):
    """Build end-to-end Scikit-learn Pipeline combining TF-IDF (1-2 grams) and engineered MetaFeatures."""
    features = ColumnTransformer([
        (
            "tfidf",
            TfidfVectorizer(
                preprocessor=clean_text,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
                sublinear_tf=True,
                max_features=max_features,
            ),
            "review",
        ),
        (
            "meta",
            make_pipeline(
                MetaFeatures(),
                SimpleImputer(strategy="median"),
                StandardScaler(),
            ),
            "review",
        ),
    ])
    return Pipeline([("features", features), ("model", model)])


def explain(review: str, pipe, top: int = 6):
    """Giải thích đóng góp của từng từ/cụm từ đối với quyết định của mô hình tuyến tính."""
    tfidf = pipe.named_steps["features"].named_transformers_["tfidf"]
    coef = pipe.named_steps["model"].coef_[0][: len(tfidf.vocabulary_)]
    v = tfidf.transform([review])
    idx = v.nonzero()[1]
    if len(idx) == 0:
        return {"negative_words": [], "positive_words": []}
    words = tfidf.get_feature_names_out()[idx]
    contrib = v.toarray()[0][idx] * coef[idx]
    order = np.argsort(contrib)
    neg_list = [(str(w), float(round(c, 3))) for w, c in zip(words[order[:top]], contrib[order[:top]]) if c < 0]
    pos_list = [(str(w), float(round(c, 3))) for w, c in zip(words[order[::-1][:top]], contrib[order[::-1][:top]]) if c > 0]
    print("Kéo về TIÊU CỰC:", neg_list)
    print("Kéo về TÍCH CỰC:", pos_list)
    return {"negative_words": neg_list, "positive_words": pos_list}

