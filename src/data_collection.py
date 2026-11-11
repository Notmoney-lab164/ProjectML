"""Script to acquire/download the IMDB 50K Movie Reviews dataset and store it in data/raw/ and data/."""
import os
import pandas as pd


def download_imdb_dataset(
    raw_path: str = "data/raw/IMDB Dataset.csv",
    main_path: str = "data/IMDB Dataset.csv",
) -> pd.DataFrame:
    """Download the 50,000 IMDB Movie Reviews dataset and save as CSV with columns ['review', 'sentiment']."""
    if os.path.exists(main_path):
        print(f"Dataset already exists at {main_path}")
        df = pd.read_csv(main_path)
        if not os.path.exists(raw_path):
            os.makedirs(os.path.dirname(raw_path), exist_ok=True)
            df.to_csv(raw_path, index=False)
        return df

    print("Downloading IMDB 50K Movie Reviews dataset via HuggingFace datasets ('stanfordnlp/imdb')...")
    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/imdb")
    train_df = pd.DataFrame(ds["train"])
    test_df = pd.DataFrame(ds["test"])
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    # Shuffle with fixed random_state=42 to match standard Kaggle IMDB 50K CSV layout
    full_df = full_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    full_df["sentiment"] = full_df["label"].map({1: "positive", 0: "negative"})
    df = full_df[["text", "sentiment"]].rename(columns={"text": "review"})

    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    os.makedirs(os.path.dirname(main_path), exist_ok=True)
    df.to_csv(raw_path, index=False)
    df.to_csv(main_path, index=False)
    print(f"Saved {len(df)} reviews to {raw_path} and {main_path}")
    return df


if __name__ == "__main__":
    download_imdb_dataset()

