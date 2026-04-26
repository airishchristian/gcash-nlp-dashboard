import os
import streamlit as st
from anthropic import Anthropic
from dotenv import load_dotenv
from utils.data_loader import load_data

load_dotenv()

st.set_page_config(
    page_title="Ask the Data | GCash",
    page_icon="💬",
    layout="wide",
)

st.title("💬 Ask the Data")
st.markdown(
    "Ask any question about the GCash review dataset. "
    "The assistant only answers questions based on the data."
)

# ── Build context summary (runs once, cached) ─────────────────────────────────
@st.cache_data(show_spinner="Preparing dataset summary...")
def build_context(_df) -> str:
    total = len(_df) - 1

    # Rating distribution
    rating_dist = (
        _df["review_rating"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    rating_str = ", ".join(
        f"{int(k)} star: {v} reviews ({v/total*100:.1f}%)"
        for k, v in rating_dist.items()
    )

    # Sentiment breakdown
    sentiment_dist = (
        _df["sentiment_meaning"]
        .value_counts()
        .to_dict()
    )
    sentiment_str = ", ".join(
        f"{k}: {v} ({v/total*100:.1f}%)"
        for k, v in sentiment_dist.items()
    )

    # Topic breakdown with avg rating
    topic_stats = (
        _df.groupby("topic")
        .agg(
            n=("review_rating", "count"),
            avg_rating=("review_rating", "mean"),
            pct_negative=(
                "sentiment_meaning",
                lambda x: (
                    x.isin(["Very negative", "Negative"]).sum()
                    / len(x) * 100
                ),
            ),
        )
        .sort_values("avg_rating")
    )
    topic_str = "\n".join(
        f"  - {row.Index}: {int(row.n)} reviews, "
        f"avg rating {row.avg_rating:.2f}, "
        f"{row.pct_negative:.1f}% negative"
        for row in topic_stats.itertuples()
    )

    # Cluster profiles
    cluster_stats = (
        _df.groupby("cluster")
        .agg(
            n=("review_rating", "count"),
            avg_rating=("review_rating", "mean"),
            avg_length=("review_length", "mean"),
            dominant_sentiment=(
                "sentiment_meaning",
                lambda x: x.mode()[0],
            ),
            top_topic=(
                "topic",
                lambda x: x.mode()[0],
            ),
        )
    )
    cluster_str = "\n".join(
        f"  - Cluster {row.Index}: {int(row.n)} reviews, "
        f"avg rating {row.avg_rating:.2f}, "
        f"avg length {row.avg_length:.0f} chars, "
        f"dominant sentiment: {row.dominant_sentiment}, "
        f"top topic: {row.top_topic}"
        for row in cluster_stats.itertuples()
    )

    # Monthly trend
    monthly = (
        _df.groupby("year_month")
        .agg(avg_rating=("review_rating", "mean"))
        .reset_index()
    )
    trend_str = ", ".join(
        f"{row.year_month}: {row.avg_rating:.2f}"
        for row in monthly.itertuples()
    )

    # Top 2 most negative reviews per topic (actual text — small sample)
    sample_reviews = []
    for topic in _df["topic"].unique():
        subset = (
            _df[
                (_df["topic"] == topic) &
                (_df["sentiment_meaning"].isin(
                    ["Very negative", "Negative"]
                ))
            ]
            .nsmallest(2, "review_rating")
            [["review_text", "review_rating"]]
        )
        for row in subset.itertuples():
            sample_reviews.append(
                f'  [{topic}, {int(row.review_rating)}★] '
                f'"{row.review_text[:120]}"'
            )
    samples_str = "\n".join(sample_reviews[:16])

    return f"""
GCASH PLAY STORE REVIEW DATASET SUMMARY
========================================
Total reviews: {total}
Date range: {_df['year_month'].min()} to {_df['year_month'].max()}

RATING DISTRIBUTION:
{rating_str}

SENTIMENT BREAKDOWN:
{sentiment_str}

TOPIC BREAKDOWN (sorted by avg rating, worst first):
{topic_str}

CUSTOMER SEGMENTS (clusters):
{cluster_str}

MONTHLY AVERAGE RATING TREND:
{trend_str}

SAMPLE NEGATIVE REVIEWS (2 per topic, truncated to 120 chars):
{samples_str}
""".strip()


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a data analyst assistant for a GCash \
Play Store review analysis project.

You have access to a summary of 1,001 GCash user reviews including \
sentiment analysis, topic classification, and customer clustering.

RULES:
1. Only answer questions about the GCash review dataset.
2. If asked something unrelated, politely say you can only discuss \
the GCash review data.
3. Be concise. Use numbers and percentages from the data.
4. When relevant, mention which topic, cluster, or sentiment class \
supports your answer.
5. Never make up data not present in the summary.

DATASET SUMMARY:
{context}"""


# ── Initialize session state ──────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "client" not in st.session_state:
    st.session_state.client = Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

# ── Load data and build context ───────────────────────────────────────────────
df = load_data()
context = build_context(df)
system = SYSTEM_PROMPT.format(context=context)

# ── Sidebar — cost tracker ────────────────────────────────────────────────────
st.sidebar.header("Session stats")

if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0

estimated_cost = st.session_state.total_tokens / 1_000_000 * 1.0
st.sidebar.metric(
    "Tokens used",
    f"{st.session_state.total_tokens:,}",
)
st.sidebar.metric(
    "Estimated cost",
    f"${estimated_cost:.4f}",
)
st.sidebar.caption(
    "Based on claude-haiku-4-5 blended rate ~$1.00/1M tokens"
)

if st.sidebar.button("Clear conversation"):
    st.session_state.messages = []
    st.session_state.total_tokens = 0
    st.rerun()

# ── Chat display ──────────────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask anything about the GCash reviews..."):

    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # Call Claude
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=512,
                system=system,
                messages=st.session_state.messages,
            )

            answer = response.content[0].text

            # Track token usage
            st.session_state.total_tokens += (
                response.usage.input_tokens
                + response.usage.output_tokens
            )

        st.markdown(answer)

    # Add assistant message to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
    })