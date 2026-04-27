import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from utils.data_loader import load_data

st.set_page_config(
    page_title="Business Insights | GCash",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Business Insights")
st.markdown("""
Six analytical questions answered from a random 1,000 GCash Play Store reviews.
All findings are derived from BERT sentiment + BART zero-shot topic classification.
""")

df = load_data()

SENTIMENT_ORDER = [
    "Very negative", "Negative", "Neutral",
    "Positive", "Very Positive"
]

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Filters")
selected_ratings = st.sidebar.multiselect(
    "Star Rating",
    options=sorted(df["review_rating"].unique()),
    default=sorted(df["review_rating"].unique()),
    format_func=lambda x: f"{int(x)} ★",
)
df_filtered = df[df["review_rating"].isin(selected_ratings)]
st.caption(f"Showing {len(df_filtered):,} of {len(df):,} reviews")

st.markdown("---")

# ── Q1: Which topics drive the most negative sentiment ────────────────────────
st.subheader("Q1 · Which topics drive the MOST negative sentiment?")
st.caption("Top topics are the biggest pain points. Sorted by % of 1-star reviews.")

topic_sentiment = (
    df_filtered
    .groupby(["topic", "sentiment"])
    .size()
    .reset_index(name="count")
)
topic_total = df_filtered.groupby("topic").size().reset_index(name="total")
topic_sentiment = topic_sentiment.merge(topic_total, on="topic")
topic_sentiment["pct"] = topic_sentiment["count"] / topic_sentiment["total"] * 100

# Sort topics by % of 1-star reviews descending (matches notebook sort_values by='1 star')
one_star_order = (
    topic_sentiment[topic_sentiment["sentiment"] == "1 star"]
    .sort_values("pct", ascending=False)["topic"]
    .tolist()
)

fig1 = px.bar(
    topic_sentiment,
    x="pct",
    y="topic",
    color="sentiment",
    orientation="h",
    barmode="stack",
    category_orders={"topic": one_star_order},
    labels={"pct": "Percentage of Reviews (%)", "topic": "", "sentiment": "Sentiment"},
)
fig1.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(fig1, use_container_width=True)

# Topics ranked by % of 1-star reviews (matches notebook print output)
st.markdown("**Topics ranked by percent of 1-star reviews:**")
one_star_table = (
    topic_sentiment[topic_sentiment["sentiment"] == "1 star"]
    [["topic", "pct"]]
    .sort_values("pct", ascending=False)
    .rename(columns={"topic": "Topic", "pct": "1-Star Reviews (%)"})
    .set_index("Topic")
    .round(2)
)
st.dataframe(one_star_table, use_container_width=True)

st.markdown("---")

# ── Q2: Topics with lowest avg star rating ────────────────────────────────────
st.subheader("Q2 · Which topics have the lowest average star rating?")
st.caption("Confirms Q1 from a different angle using the user's actual numeric rating.")

avg = (
    df_filtered
    .groupby("topic")
    .agg(
        avg_rating=("review_rating", "mean"),
        n_reviews=("review_rating", "count"),
        avg_sentiment_score=("sentiment_score", "mean"),
    )
    .sort_values("avg_rating", ascending=False)
    .reset_index()
    .round(2)
)
st.dataframe(avg.set_index("topic"), use_container_width=True)

st.markdown("---")

# ── Q3: Cluster profiles ──────────────────────────────────────────────────────
st.subheader("Q3 · Cluster profiles — who are these customer segments?")
st.caption(
    "A cluster of frustrated long-review writers needs different treatment "
    "than a cluster of casual one-liner reviewers."
)

for cluster_id in sorted(df_filtered["cluster"].unique()):
    cluster_df = df_filtered[df_filtered["cluster"] == cluster_id]
    pct = len(cluster_df) / len(df_filtered) * 100
    with st.expander(f"Cluster {cluster_id} — {len(cluster_df):,} reviews ({pct:.1f}%)"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Rating", f"{cluster_df['review_rating'].mean():.2f} ★")
        col2.metric("Avg Sentiment Score", f"{cluster_df['sentiment_score'].mean():.2f}")
        col3.metric("Avg Review Length", f"{cluster_df['review_length'].mean():.0f} chars")
        st.write(f"**Dominant sentiment:** {cluster_df['sentiment'].mode()[0]}")
        st.write(f"**Top topic:** {cluster_df['topic'].mode()[0]}")

st.markdown("---")

# ── Q4: Sentiment over time ───────────────────────────────────────────────────
st.subheader("Q4 · Time trends — is sentiment improving or getting worse?")
st.caption("After a product change or incident, did customer sentiment shift?")

df_filtered = df_filtered.copy()
df_filtered["year_month"] = pd.to_datetime(df_filtered["review_datetime_utc"]).dt.strftime('%Y-%m')

monthly = (
    df_filtered
    .groupby("year_month")
    .agg(
        avg_rating=("review_rating", "mean"),
        avg_sentiment=("sentiment_score", "mean"),
        n_reviews=("review_datetime_utc", "count"),
    )
    .reset_index()
)
monthly = monthly[monthly["n_reviews"] >= 5]

fig4 = px.line(
    monthly,
    x="year_month",
    y="avg_rating",
    markers=True,
    labels={"year_month": "Month", "avg_rating": "Avg Rating"},
    color_discrete_sequence=["#007dfe"],
    title="GCash — Monthly Rating Trend",
)
fig4.add_hline(
    y=df_filtered["review_rating"].mean(),
    line_dash="dash",
    line_color="gray",
    annotation_text="Overall avg",
)
st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# ── Q5: App version impact ────────────────────────────────────────────────────
st.subheader("Q5 · Which app versions generate the most complaints?")
st.caption("Versions below the overall average are red flags for engineering.")

version_counts = df_filtered["author_app_version"].value_counts()
valid_versions = version_counts[version_counts >= 5].index

if len(valid_versions) > 0:
    version_stats = (
        df_filtered[df_filtered["author_app_version"].isin(valid_versions)]
        .groupby("author_app_version")
        .agg(
            avg_rating=("review_rating", "mean"),
            n_reviews=("review_rating", "count"),
        )
        .sort_values("avg_rating")
        .head(15)
        .reset_index()
        .round(2)
    )

    print("Bottom 15 worst-rated app versions (min 5 reviews):\n")

    fig5 = px.bar(
        version_stats,
        x="avg_rating",
        y="author_app_version",
        orientation="h",
        color="avg_rating",
        color_continuous_scale="RdYlGn",
        labels={
            "avg_rating": "Avg ★",
            "author_app_version": "App Version",
        },
        text=version_stats["avg_rating"].round(2),
    )
    fig5.update_layout(
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig5, use_container_width=True)
else:
    st.info("Not enough reviews per version for meaningful analysis.")

st.markdown("---")

# ── Q6: Model limitations ─────────────────────────────────────────────────────
st.subheader("Q6 · Where does the model struggle?")
st.caption(
    "Reviews with low confidence scores deserve manual review "
    "regardless of their assigned label."
)

TOPIC_THRESHOLD = 0.65
SENTIMENT_THRESHOLD = 0.65

uncertain_topic = df_filtered[df_filtered["topic_confidence"] < TOPIC_THRESHOLD]
uncertain_sentiment = df_filtered[df_filtered["sentiment_score"] < SENTIMENT_THRESHOLD]

c1, c2 = st.columns(2)
with c1:
    st.metric(
        "Low-confidence topic classifications",
        f"{len(uncertain_topic):,}",
        f"{len(uncertain_topic)/len(df_filtered)*100:.1f}% of reviews",
    )
    st.caption(
        "These reviews had weak signal for all 8 topics. "
        "Consider adding more topic categories."
    )

with c2:
    st.metric(
        "Uncertain sentiment classifications",
        f"{len(uncertain_sentiment):,}",
        f"{len(uncertain_sentiment)/len(df_filtered)*100:.1f}% of reviews",
    )
    st.caption(
        "Neutral tone, factual complaints, or Taglish sarcasm "
        "confuse the multilingual BERT model."
    )

# Confidence score distributions
fig6a = px.histogram(
    df_filtered, x="topic_confidence", nbins=30,
    title="Topic Confidence Score Distribution",
    labels={"topic_confidence": "Confidence"},
    color_discrete_sequence=["#007dfe"],
)
fig6a.add_vline(x=TOPIC_THRESHOLD, line_dash="dash", line_color="red",
                annotation_text=f"Threshold ({TOPIC_THRESHOLD})")

fig6b = px.histogram(
    df_filtered, x="sentiment_score", nbins=30,
    title="Sentiment Confidence Score Distribution",
    labels={"sentiment_score": "Confidence"},
    color_discrete_sequence=["#ff7f0e"],
)
fig6b.add_vline(x=SENTIMENT_THRESHOLD, line_dash="dash", line_color="red",
                annotation_text=f"Threshold ({SENTIMENT_THRESHOLD})")

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig6a, use_container_width=True)
with col2:
    st.plotly_chart(fig6b, use_container_width=True)

# Sample low-confidence reviews for manual inspection
with st.expander("🔍 Sample low-confidence topic reviews (manual inspection)"):
    st.dataframe(
        uncertain_topic[["review_text", "topic", "topic_confidence", "sentiment_meaning"]]
        .sort_values("topic_confidence")
        .head(10)
        .reset_index(drop=True),
        use_container_width=True,
    )

st.markdown("---")

# ── Recommendations ───────────────────────────────────────────────────────────
st.subheader("Recommendations")

worst_topic = (
    df_filtered[df_filtered["sentiment"].isin(["1 star", "2 stars"])]
    .groupby("topic")
    .size()
    .idxmax()
)

st.error(f"🔴 **Fix first:** {worst_topic} — highest volume of negative reviews.")
st.warning(
    "🟡 **Monitor** monthly rating trend — any dip below the overall "
    "average after a release is a signal to investigate."
)
st.info(
    "🔵 **Segment** your response strategy by cluster — "
    "long frustrated reviews need detailed support responses, "
    "not templated replies."
)
st.write(
    "⚪ **Improve model** — low-confidence reviews need manual labelling "
    "or an expanded topic taxonomy."
)