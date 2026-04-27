import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/gcash_reviews_enriched.csv"


@st.cache_data(show_spinner="Loading dataset...")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    df["review_datetime_utc"] = pd.to_datetime(
        df["review_datetime_utc"], errors="coerce"
    )

    if "review_length" not in df.columns:
        df["review_length"] = df["review_text"].str.len()
    if "word_count" not in df.columns:
        df["word_count"] = df["review_text"].str.split().str.len()
    if "year_month" not in df.columns:
        df["year_month"] = df["review_datetime_utc"].dt.strftime("%Y-%m")

    return df


# def _add_pca(df: pd.DataFrame) -> pd.DataFrame:
#     if "pca_x" in df.columns and "pca_y" in df.columns:
#         return df

#     features = ["review_rating", "sentiment_score",
#                 "review_length", "word_count"]

#     X = df[features].dropna()
#     X_scaled = StandardScaler().fit_transform(X)
#     coords = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

#     df = df.copy()
#     df.loc[X.index, "pca_x"] = coords[:, 0]
#     df.loc[X.index, "pca_y"] = coords[:, 1]

#     return df