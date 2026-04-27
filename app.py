import streamlit as st

st.set_page_config(
    page_title="GCash Review Intelligence",
    page_icon="💙",
    layout="wide",
)

st.title("💙 GCash Google Play Review Intelligence Dashboard")

st.markdown("""
Welcome. This dashboard turns GCash Play Store reviews
into actionable product insights using NLP.

# GCash App Reviews — End-to-End NLP Pipeline

## Why This Dataset is Great
- **Philippine-based:** GCash is the #1 e-wallet in the Philippines (owned by Globe Telecom)
- **Real Filipino users:** Reviews include English, Tagalog, and Taglish — authentic voice of Filipino customers
- **Large volume:** Hundreds of thousands of reviews, plenty to analyze
- **Service-industry relevant:** The pain points (app crashes, customer support, payment failures, account issues) are the same types of issues any service company — including telecom — faces


## The Business Question
**If I were on the GCash product team, what would I fix first?** By the end of this notebook, we'll have data-driven answers.
""")

st.info("This dataset was downloaded from Kaggle @bwandowando/globe-gcash-google-app-reviews")