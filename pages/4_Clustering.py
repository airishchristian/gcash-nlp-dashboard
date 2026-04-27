import streamlit as st
import plotly.express as px
from utils.data_loader import load_data
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Clustering | GCash",
    page_icon="🔵",
    layout="wide",
)

st.title("🔵 Customer Segmentation")
st.markdown("""
Reviews clustered using **PCA + KMeans** on features:
`review_rating`, `sentiment_score`, `review_length`, `word_count`, `sentiment`.  
Each cluster represents a distinct type of GCash reviewer.
""")

df = load_data()

if "pca_x" not in df.columns and "pca_y" not in df.columns:

    features = ["review_rating", "sentiment_score",
                "review_length", "word_count"]

    X = df[features].dropna()
    X_scaled = StandardScaler().fit_transform(X)
    coords = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

    df = df.copy()
    df.loc[X.index, "pca_x"] = coords[:, 0]
    df.loc[X.index, "pca_y"] = coords[:, 1]

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.header("Filters")

selected_clusters = st.sidebar.multiselect(
    "Cluster",
    options=sorted(df["cluster"].unique()),
    default=sorted(df["cluster"].unique()),
    format_func=lambda x: f"Cluster {x}",
)

df_filtered = df[df["cluster"].isin(selected_clusters)]

st.caption(f"Showing {len(df_filtered):,} of {len(df):,} reviews")

# ── KPIs ──────────────────────────────────────────────────────────────────────
st.markdown("---")
k1, k2, k3 = st.columns(3)
k1.metric("Total Reviews", f"{len(df_filtered):,}")
k2.metric("Clusters", f"{df_filtered['cluster'].nunique()}")
k3.metric("Avg Rating", f"{df_filtered['review_rating'].mean():.2f} ★")

# ── Cluster scatter ───────────────────────────────────────────────────────────
st.subheader("Cluster map")
st.caption("Each dot is one review. Color = cluster.")

if "pca_x" in df.columns and "pca_y" in df.columns:
    fig = px.scatter(
        df_filtered,
        x="pca_x",
        y="pca_y",
        color=df_filtered["cluster"].astype(str),
        hover_data=["review_rating", "sentiment_meaning",
                    "topic", "review_text"],
        labels={"pca_x": "PC1", "pca_y": "PC2", "color": "Cluster"},
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info(
        "PCA coordinates not found in dataset. "
        "Re-run the notebook with PCA export enabled."
    )

# ── Cluster profiles ──────────────────────────────────────────────────────────
st.subheader("Cluster profiles")

profile_cols = ["review_rating", "sentiment_score",
                "review_length", "word_count"]

profile = (
    df_filtered
    .groupby("cluster")[profile_cols]
    .mean()
    .round(2)
)
profile["n_reviews"] = df_filtered.groupby("cluster").size()
profile["dominant_sentiment"] = (
    df_filtered.groupby("cluster")["sentiment_meaning"]
    .agg(lambda x: x.mode()[0])
)
profile["top_topic"] = (
    df_filtered.groupby("cluster")["topic"]
    .agg(lambda x: x.mode()[0])
)

st.dataframe(profile, use_container_width=True)

# ── Per cluster drill down ────────────────────────────────────────────────────
st.subheader("Drill down by cluster")

cluster_choice = st.selectbox(
    "Select a cluster",
    options=sorted(df_filtered["cluster"].unique()),
    format_func=lambda x: f"Cluster {x}",
)

cluster_df = df_filtered[df_filtered["cluster"] == cluster_choice]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews", f"{len(cluster_df):,}")
c2.metric("Avg Rating", f"{cluster_df['review_rating'].mean():.2f} ★")
c3.metric("Avg Sentiment Score",
          f"{cluster_df['sentiment_score'].mean():.2f}")
c4.metric("Dominant Sentiment",
          cluster_df["sentiment_meaning"].mode()[0])

st.markdown(f"**Top topic:** {cluster_df['topic'].mode()[0]}")

st.dataframe(
    cluster_df[["review_text", "review_rating",
                "sentiment_meaning", "topic"]]
    .sample(min(10, len(cluster_df)), random_state=42),
    use_container_width=True,
    hide_index=True,
)