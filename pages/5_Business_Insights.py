import streamlit as st
import plotly.express as px
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

# ── Q1: Topics driving most negative sentiment ────────────────────────────────
st.subheader("Q1 · Which topics drive the most negative sentiment?")
st.caption("The longer the red bar, the bigger the product pain point.")

ct = (
    df_filtered
    .groupby(["topic", "sentiment_meaning"])
    .size()
    .reset_index(name="count")
    .sort_index()
)
fig1 = px.bar(
    ct,
    x="count",
    y="topic",
    color="sentiment_meaning",
    orientation="h",
    barmode="stack",
    category_orders={"sentiment_meaning": SENTIMENT_ORDER},
    color_discrete_sequence=[
        "#d62728", "#ff7f0e", "#bcbd22", "#2ca02c", "#1f77b4"
    ],
    labels={"count": "Reviews", "topic": "",
            "sentiment_meaning": "Sentiment"},
)
fig1.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(fig1, use_container_width=True)

# key finding callout
worst_topic = (
    df_filtered[df_filtered["sentiment_meaning"]
                .isin(["Very negative", "Negative"])]
    .groupby("topic")
    .size()
    .idxmax()
)
st.error(f"Biggest pain point: **{worst_topic}**")

st.markdown("---")

# ── Q2: Topics with lowest avg star rating ────────────────────────────────────
st.subheader("Q2 · Which topics have the lowest average star rating?")
st.caption("Confirms Q1 from a different angle using the user's own numeric score.")

avg = (
    df_filtered
    .groupby("topic")
    .agg(
        avg_rating=("review_rating", "mean"),
        n_reviews=("review_rating", "count"),
    )
    .sort_values("avg_rating")
    .reset_index()
)
fig2 = px.bar(
    avg,
    x="avg_rating",
    y="topic",
    orientation="h",
    color="avg_rating",
    color_continuous_scale="RdYlGn",
    labels={"avg_rating": "Avg ★", "topic": ""},
    text=avg["avg_rating"].round(2),
)
fig2.update_layout(
    yaxis=dict(autorange="reversed"),
    coloraxis_showscale=False,
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ── Q3: Cluster profiles ──────────────────────────────────────────────────────
st.subheader("Q3 · Who are the customer segments?")
st.caption(
    "Long reviews + low ratings = vocal dissatisfied users. "
    "Short reviews + high ratings = satisfied casual users."
)

profile = (
    df_filtered
    .groupby("cluster")
    .agg(
        avg_rating=("review_rating", "mean"),
        avg_length=("review_length", "mean"),
        avg_sentiment=("sentiment_score", "mean"),
        n_reviews=("review_rating", "count"),
    )
    .round(2)
)
profile["dominant_sentiment"] = (
    df_filtered.groupby("cluster")["sentiment_meaning"]
    .agg(lambda x: x.mode()[0])
)
profile["top_topic"] = (
    df_filtered.groupby("cluster")["topic"]
    .agg(lambda x: x.mode()[0])
)

st.dataframe(profile, use_container_width=True)

st.markdown("---")

# ── Q4: Sentiment over time ───────────────────────────────────────────────────
st.subheader("Q4 · Is sentiment improving or getting worse over time?")
st.caption("Watch for dips after major app releases or incidents.")

monthly = (
    df_filtered
    .groupby("year_month")
    .agg(
        avg_rating=("review_rating", "mean"),
        n_reviews=("review_rating", "count"),
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
        df_filtered[df_filtered["author_app_version"]
                    .isin(valid_versions)]
        .groupby("author_app_version")
        .agg(
            avg_rating=("review_rating", "mean"),
            n_reviews=("review_rating", "count"),
        )
        .sort_values("avg_rating")
        .head(15)
        .reset_index()
    )
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

c1, c2 = st.columns(2)

with c1:
    uncertain = df_filtered[df_filtered["topic_confidence"] < 0.65]
    st.metric(
        "Low-confidence topic classifications",
        f"{len(uncertain):,}",
        f"{len(uncertain)/len(df_filtered)*100:.1f}% of reviews",
    )
    st.caption(
        "These reviews had weak signal for all 8 topics. "
        "Consider adding more topic categories."
    )

with c2:
    low_sentiment = df_filtered[df_filtered["sentiment_score"] < 0.65]
    st.metric(
        "Uncertain sentiment classifications",
        f"{len(low_sentiment):,}",
        f"{len(low_sentiment)/len(df_filtered)*100:.1f}% of reviews",
    )
    st.caption(
        "Neutral tone, factual complaints, or Taglish sarcasm "
        "confuse the multilingual BERT model."
    )

st.markdown("---")

# ── Recommendations ───────────────────────────────────────────────────────────
st.subheader("Recommendations")

st.success(f"Fix first: **{worst_topic}** — highest volume of negative reviews.")
st.warning(
    "Monitor monthly rating trend — any dip below the overall "
    "average after a release is a signal to investigate."
)
st.info(
    "Segment your response strategy by cluster — "
    "long frustrated reviews need detailed support responses, "
    "not templated replies."
)