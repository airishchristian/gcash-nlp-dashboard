import streamlit as st
import plotly.express as px
from utils.data_loader import load_data

st.set_page_config(
    page_title="Sentiment | GCash",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Sentiment Analysis")
st.markdown("""
Model: `nlptown/bert-base-multilingual-uncased-sentiment`  
Each review is classified into 5 sentiment levels — from Very Negative to Very Positive.
""")

df = load_data()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.header("Filters")

selected_ratings = st.sidebar.multiselect(
    "Star Rating",
    options=sorted(df["review_rating"].unique()),
    default=sorted(df["review_rating"].unique()),
    format_func=lambda x: f"{int(x)} ★",
)

SENTIMENT_ORDER = [
    "Very negative", "Negative", "Neutral", "Positive", "Very Positive"
]

selected_sentiments = st.sidebar.multiselect(
    "Sentiment",
    options=SENTIMENT_ORDER,
    default=SENTIMENT_ORDER,
)

df_filtered = df[
    df["review_rating"].isin(selected_ratings) &
    df["sentiment_meaning"].isin(selected_sentiments)
]

st.caption(f"Showing {len(df_filtered):,} of {len(df):,} reviews")

# ── KPIs ──────────────────────────────────────────────────────────────────────
st.markdown("---")
total = len(df_filtered)

neg = df_filtered["sentiment_meaning"].isin(
    ["Very negative", "Negative"]).sum()
neu = df_filtered["sentiment_meaning"].eq("Neutral").sum()
pos = df_filtered["sentiment_meaning"].isin(
    ["Positive", "Very Positive"]).sum()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Reviews", f"{total:,}")
k2.metric("Negative", f"{neg:,}", f"{neg/total*100:.1f}%")
k3.metric("Neutral", f"{neu:,}", f"{neu/total*100:.1f}%")
k4.metric("Positive", f"{pos:,}", f"{pos/total*100:.1f}%")

# ── Charts ────────────────────────────────────────────────────────────────────
SENTIMENT_COLORS = {
    "Very negative": "#d62728",
    "Negative":      "#ff7f0e",
    "Neutral":       "#bcbd22",
    "Positive":      "#2ca02c",
    "Very Positive": "#1f77b4",
}

c1, c2 = st.columns(2)

with c1:
    st.subheader("Overall sentiment distribution")
    counts = (
        df_filtered["sentiment_meaning"]
        .value_counts()
        .reindex(SENTIMENT_ORDER, fill_value=0)
        .reset_index()
    )
    counts.columns = ["sentiment", "count"]
    fig = px.bar(
        counts,
        x="sentiment",
        y="count",
        color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        labels={"sentiment": "", "count": "Reviews"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Sentiment by star rating")
    ct = (
        df_filtered
        .groupby(["review_rating", "sentiment_meaning"])
        .size()
        .reset_index(name="count")
    )
    fig2 = px.bar(
        ct,
        x="review_rating",
        y="count",
        color="sentiment_meaning",
        barmode="stack",
        color_discrete_map=SENTIMENT_COLORS,
        labels={
            "review_rating": "Star Rating",
            "count": "Reviews",
            "sentiment_meaning": "Sentiment",
        },
        category_orders={"sentiment_meaning": SENTIMENT_ORDER},
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Sample reviews ────────────────────────────────────────────────────────────
st.subheader("Sample reviews by sentiment")

sentiment_choice = st.selectbox(
    "Pick a sentiment class to explore",
    options=SENTIMENT_ORDER,
)

sample = (
    df_filtered[df_filtered["sentiment_meaning"] == sentiment_choice]
    [["review_text", "review_rating", "sentiment_score"]]
    .sample(min(5, len(df_filtered)), random_state=42)
)

st.dataframe(sample, use_container_width=True, hide_index=True)