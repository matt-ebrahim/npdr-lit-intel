# Clinical AI Literature Tracker

A Streamlit application for searching, filtering, and extracting structured data from clinical AI/ML literature across any indication.

## Features

- **MeSH-based search** via snek ontology mapping
- **Dynamic challenge classification** - describe your research needs, LLM maps to standard categories
- **Structured extraction** - extracts methodology, metrics, and challenge relevance from papers
- **Quality filtering** - journal tier and relevance scoring
- **Visualizations** - publication trends, journal distribution, challenge coverage

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env  # Add your API keys

# Run the app
streamlit run npdr_pipeline/app.py
```

## Configuration

Required environment variables in `.env`:
- `LITELLM_API_KEY` - LLM gateway key
- `SNEK_API_KEY` - Ontology mapping API
- Snowflake credentials for PubMed access

## Standard Challenges

The pipeline assesses papers against 6 clinical AI challenges:
- Long-term Prediction
- Early Detection
- Class Imbalance / Low Event Rates
- Rapid Progressors
- Diagnostic Consistency
- Risk Stratification
