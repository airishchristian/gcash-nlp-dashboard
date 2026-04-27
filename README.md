# GCash Review using NLP Dashboard

> Turning 1,00 Google Play Store reviews into actionable product insights using NLP.

A fully interactive, multi-page Streamlit dashboard that applies BERT sentiment analysis, zero-shot topic classification, and KMeans clustering to real GCash user reviews — with a built-in Claude AI analyst you can query in natural language.

---

## Live Demo

🔗 [gcash-nlp-dashboard.streamlit.app](#) ← _replace with your URL after deployment_

---

## Key Findings

- **App Crashes & Bugs** is the lowest-rated topic (avg ★ below overall average) and drives the highest volume of negative reviews — the clearest priority for the engineering team
- **Long reviews correlate with frustration** — the cluster with the longest average review length also has the lowest average star rating
- **Sentiment confidence below 0.65** often signals factual complaints written in neutral tone or Taglish sarcasm — not model errors, but genuine uncertainty
- Monthly rating trends reveal dips that align with specific app version releases

---

## What This Project Does

The dashboard is built on top of a pre-enriched dataset produced by a companion Jupyter notebook (`01_eda.ipynb`). It presents six layers of analysis across five dedicated pages plus an AI chat interface:

| Page | What it shows |
|---|---|
| EDA | Rating distribution, monthly trends, review length analysis |
| Sentiment Analysis | BERT-classified sentiment across 5 levels, breakdown by star rating |
| Topic Classification | 8 zero-shot business topics, sentiment mix per topic |
| Customer Segmentation | PCA + KMeans clusters, cluster profiles and drill-down |
| Business Insights | Six analytical questions answered with charts and recommendations |
| Ask the Data | Claude-powered chat interface — ask anything about the dataset |

---

## NLP Pipeline

### Sentiment Analysis
**Model:** `nlptown/bert-base-multilingual-uncased-sentiment`

A multilingual BERT model fine-tuned on product reviews. Classifies each review into five sentiment levels from Very Negative to Very Positive. Works on English and Taglish, though neutral-toned complaints and sarcasm reduce confidence scores.

### Topic Classification
**Model:** `facebook/bart-large-mnli` (zero-shot)

No fine-tuning required. Each review is classified against eight GCash-specific business topics using natural language inference:

- App Crashes or Bugs
- Account Login Issues
- Money Transfer and Cash In
- Payment and Bills
- Customer Service Support
- Fees and Charges
- Security and Fraud
- User Interface and Experience

### Customer Segmentation
**Method:** PCA (2 components) + KMeans (4 clusters)

Features: `review_rating`, `sentiment_score`, `review_length`, `word_count`, `sentiment` (one-hot encoded). PCA reduces dimensionality for visualization; KMeans assigns segment labels.

### AI Chat Interface
**Model:** `claude-haiku-4-5`

The chat page sends a precomputed statistical summary (~800 tokens) as context instead of raw review text — keeping cost to ~$0.001 per message while still grounding answers in the actual data. The assistant is scoped to only answer questions about the GCash dataset.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Dashboard | Streamlit 1.56+ |
| Charts | Plotly Express |
| Data | pandas, numpy |
| NLP | Hugging Face Transformers |
| ML | scikit-learn |
| AI Chat | Anthropic Claude API |
| Package management | uv |
| Deployment | Streamlit Community Cloud |

---

## Project Structure

```
gcash-nlp-dashboard/
├── app.py                        ← entry point / home page
├── pages/
│   ├── 1_EDA.py                  ← exploratory data analysis
│   ├── 2_Sentiment.py            ← BERT sentiment results
│   ├── 3_Topics.py               ← zero-shot topic classification
│   ├── 4_Clustering.py           ← PCA + KMeans segmentation
│   ├── 5_Business_Insights.py    ← six analytical questions
│   └── 6_Ask_the_Data.py         ← Claude AI chat interface
├── utils/
│   └── data_loader.py            ← cached data loading + PCA
├── components/
│   ├── charts.py                 ← reusable Plotly helpers
│   └── filters.py                ← sidebar filter widgets
├── .streamlit/
│   └── config.toml               ← theme (GCash blue #007dfe)
├── pyproject.toml                ← uv dependency manifest
├── requirements.txt              ← Streamlit Cloud build file
└── .env                          ← API keys (never committed)
```

---

## Running Locally

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) installed
- Anthropic API key

### Setup

```bash
# Clone the repo
git clone https://github.com/<you>/gcash-nlp-dashboard.git
cd gcash-nlp-dashboard

# Install dependencies
uv sync

# Add your API key
echo "ANTHROPIC_API_KEY=your_key_here" > .env

# Place your enriched dataset
mkdir data
cp /path/to/gcash_reviews_enriched.csv data/

# Run the app
uv run streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Dataset

**Source:** GCash Google Play Store Reviews via [Kaggle](https://www.kaggle.com/datasets/bwandowando/globe-gcash-google-app-reviews)

**Sample:** 1,001 reviews stratified by star rating

**Enriched columns produced by the notebook:**

| Column | Description |
|---|---|
| `sentiment` | BERT raw label (1–5 stars) |
| `sentiment_score` | Model confidence (0–1) |
| `sentiment_meaning` | Human-readable label |
| `topic` | Zero-shot business topic |
| `topic_confidence` | BART confidence score |
| `cluster` | KMeans cluster ID |
| `review_length` | Character count |
| `word_count` | Word count |
| `year_month` | Formatted date for time series |

The dataset is not committed to this repository. Run `01_eda.ipynb` to reproduce it.

---

## Deployment

This app is deployed on [Streamlit Community Cloud](https://streamlit.io/cloud).

For the `ANTHROPIC_API_KEY` secret, add it via the Streamlit Cloud dashboard under **Settings → Secrets**:

```toml
ANTHROPIC_API_KEY = "your_key_here"
```

The `data/` folder is not committed. After deployment, the app expects the enriched CSV to be available — see the deployment section of the companion notebook for options.

---

## Author

Built by **[Your Name]** as part of an NLP portfolio project.

[LinkedIn](#) · [GitHub](#) · [Email](#)