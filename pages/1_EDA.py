import streamlit as st
import plotly.express as px
from utils.data_loader import load_data

st.set_page_config(
    page_title="EDA | GCash",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Exploratory Data Analysis")
st.markdown("A first look at the GCash review dataset before any NLP.")

df = load_data()

# ── KPI row ───────────────────────────────────────────────────────────────────
st.markdown("---")
k1, k2, k3, k4 = st.columns(4)

k1.metric("Total Reviews", f"{len(df):,}")
k2.metric("Avg Rating", f"{df['review_rating'].mean():.2f} ★")
k3.metric("1-Star Reviews", f"{(df['review_rating'] == 1).sum():,}")
k4.metric("5-Star Reviews", f"{(df['review_rating'] == 5).sum():,}")

# ── Sidebar filter ────────────────────────────────────────────────────────────
st.sidebar.header("Filters")
selected_ratings = st.sidebar.multiselect(
    "Star Rating",
    options=sorted(df["review_rating"].unique()),
    default=sorted(df["review_rating"].unique()),
    format_func=lambda x: f"{int(x)} ★",
)

df_filtered = df[df["review_rating"].isin(selected_ratings)]

st.caption(f"Showing {len(df_filtered):,} of {len(df):,} reviews")

# ── Charts ────────────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.subheader("Rating distribution")
    fig = px.bar(
        df_filtered["review_rating"]
            .value_counts()
            .sort_index()
            .reset_index(),
        x="review_rating",
        y="count",
        labels={"review_rating": "Star Rating", "count": "Reviews"},
        color_discrete_sequence=["#007dfe"],
    )
    fig.update_layout(xaxis=dict(tickmode="linear", dtick=1))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Reviews over time")
    monthly = (
        df_filtered
        .groupby("year_month")
        .agg(reviews=("review_rating", "count"),
             avg_rating=("review_rating", "mean"))
        .reset_index()
    )
    fig2 = px.line(
        monthly,
        x="year_month",
        y="avg_rating",
        markers=True,
        labels={"year_month": "Month", "avg_rating": "Avg Rating"},
        color_discrete_sequence=["#007dfe"],
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Review length distribution ────────────────────────────────────────────────
st.subheader("Review length distribution")
fig3 = px.histogram(
    df_filtered,
    x="review_length",
    nbins=50,
    labels={"review_length": "Characters"},
    color_discrete_sequence=["#007dfe"],
)
st.plotly_chart(fig3, use_container_width=True)

# ── Raw data explorer ─────────────────────────────────────────────────────────
with st.expander("Browse raw reviews"):
    st.dataframe(
        df_filtered[["review_text", "review_rating",
                     "review_datetime_utc", "author_app_version"]],
        use_container_width=True,
        hide_index=True,
    )