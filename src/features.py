import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

NEGATIONS = {"not", "no", "never", "nor", "cannot", "n't"}


def clean_text(s: str) -> str:
    """Clean raw HTML tags and normalize whitespace while preserving negation words."""
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    s = re.sub(r"<.*?>", " ", s)        # bỏ thẻ HTML <br />
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()            # KHÔNG xóa từ phủ định ("not good" khác "good")


class MetaFeatures(BaseEstimator, TransformerMixin):
    """Feature tự tạo, tính trên text gốc (trước khi lowercase).

    Trích xuất 4 đặc trưng:
    1. word_count: Tổng số từ trong review
    2. exclamation_count: Số dấu chấm than (!)
    3. uppercase_ratio: Tỷ lệ ký tự viết hoa trên tổng độ dài chuỗi
    4. negation_count: Số lượng từ phủ định (not, no, never, nor, cannot, n't)
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        X = pd.Series(X).fillna("").astype(str)
        words = X.str.split()
        return np.c_[
            words.str.len(),                                                  # số từ
            X.str.count("!"),                                                 # số dấu !
            X.apply(lambda t: sum(c.isupper() for c in t) / max(len(t), 1)),  # tỷ lệ chữ in hoa
            words.apply(
                lambda w: sum(
                    t.lower().strip(".,!?\"'()[]") in NEGATIONS or t.lower().endswith("n't")
                    for t in w
                )
            ),                                                                # số từ phủ định
        ]

    def get_feature_names_out(self, input_features=None):
        return np.array(["word_count", "exclamation_count", "uppercase_ratio", "negation_count"])

