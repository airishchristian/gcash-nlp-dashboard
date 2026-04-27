import streamlit as st
import plotly.express as px
from utils.data_loader import load_data

st.set_page_config(
    page_title="Topics | GCash",
    page_icon="🗂️",
    layout="wide",
)

st.title("🗂️ Topic Classification")
st.markdown("""
Model: `facebook/bart-large-mnli` (zero-shot)  
Each review is assigned to one of 8 GCash-specific business topics
without any fine-tuning.
""")

df = load_data()

SENTIMENT_ORDER = [
    "Very negative", "Negative", "Neutral", "Positive", "Very Positive"
]

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.header("Filters")

selected_ratings = st.sidebar.multiselect(
    "Star Rating",
    options=sorted(df["review_rating"].unique()),
    default=sorted(df["review_rating"].unique()),
    format_func=lambda x: f"{int(x)} ★",
)

selected_topics = st.sidebar.multiselect(
    "Topic",
    options=sorted(df["topic"].unique()),
    default=sorted(df["topic"].unique()),
)

df_filtered = df[
    df["review_rating"].isin(selected_ratings) &
    df["topic"].isin(selected_topics)
]

st.caption(f"Showing {len(df_filtered):,} of {len(df):,} reviews")

# ── KPIs ──────────────────────────────────────────────────────────────────────
st.markdown("---")
k1, k2, k3 = st.columns(3)

k1.metric("Total Reviews", f"{len(df_filtered):,}")
k2.metric("Topics Detected", f"{df_filtered['topic'].nunique()}")
k3.metric(
    "Avg Topic Confidence",
    f"{df_filtered['topic_confidence'].mean():.2f}",
)

# ── Charts ────────────────────────────────────────────────────────────────────
st.subheader("Topic distribution")
topic_counts = (
    df_filtered["topic"]
    .value_counts()
    .reset_index()
)
topic_counts.columns = ["topic", "count"]

fig = px.bar(
    topic_counts,
    x="count",
    y="topic",
    orientation="h",
    color_discrete_sequence=["#007dfe"],
    labels={"count": "Number of Reviews", "topic": ""},
)
fig.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(fig, use_container_width=True)

# ── Sentiment mix per topic ───────────────────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.subheader("Sentiment mix by topic")
    ct = (
        df_filtered
        .groupby(["topic", "sentiment_meaning"])
        .size()
        .reset_index(name="count")
    )
    fig2 = px.bar(
        ct,
        x="count",
        y="topic",
        color="sentiment_meaning",
        orientation="h",
        barmode="stack",
        labels={
            "count": "Reviews",
            "topic": "",
            "sentiment_meaning": "Sentiment",
        },
        category_orders={"sentiment_meaning": SENTIMENT_ORDER},
    )
    fig2.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig2, use_container_width=True)

with c2:
    st.subheader("Avg rating by topic")
    avg_rating = (
        df_filtered
        .groupby("topic")
        .agg(
            avg_rating=("review_rating", "mean"),
            n_reviews=("review_rating", "count"),
        )
        .sort_values("avg_rating")
        .reset_index()
    )
    fig3 = px.bar(
        avg_rating,
        x="avg_rating",
        y="topic",
        orientation="h",
        color="avg_rating",
        color_continuous_scale="RdYlGn",
        labels={"avg_rating": "Avg ★", "topic": ""},
    )
    fig3.update_layout(
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── Drill down ────────────────────────────────────────────────────────────────
st.subheader("Drill down into a topic")

topic_choice = st.selectbox(
    "Select a topic",
    options=sorted(df_filtered["topic"].unique()),
)

topic_df = df_filtered[df_filtered["topic"] == topic_choice]

t1, t2, t3 = st.columns(3)
t1.metric("Reviews", f"{len(topic_df):,}")
t2.metric("Avg Rating", f"{topic_df['review_rating'].mean():.2f} ★")
t3.metric(
    "Avg Confidence",
    f"{topic_df['topic_confidence'].mean():.2f}",
)

st.dataframe(
    topic_df[["review_text", "review_rating",
              "sentiment_meaning", "topic_confidence"]]
    .sort_values("topic_confidence", ascending=False),
    use_container_width=True,
    hide_index=True,
)