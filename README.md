# AI-Powered Customer Churn Analytics Dashboard

This project combines customer churn analytics, interactive data visualization, and a Gemini-powered assistant into a single Streamlit application. Users can explore churn trends through a dashboard, filter the customer base in real time, and ask natural-language questions about the data, model insights, and chart behavior.

## Project Architecture

```text
Dataset
  -> Data Cleaning
  -> Exploratory Data Analysis
  -> Machine Learning Model
  -> Feature Importance Extraction
  -> Insights JSON
  -> Power BI Dashboard
  -> Streamlit Web App
  -> Gemini Insight Assistant
```

## What The Project Includes

- Cleaned `BankChurners` dataset with churn encoded for modeling and analysis
- Exploratory analysis of churn, demographics, transactions, inactivity, and utilization
- Random Forest churn model with extracted churn drivers
- Structured `churn_insights.json` used by the web app
- Dark-themed Streamlit dashboard with sidebar filters, KPI cards, summary tables, and Plotly charts
- Gemini-powered Ask AI assistant with dashboard-aware fallback responses
- Power BI report section with embed support when an embed URL is available

## Dashboard Features

- KPI header for total customers, churned customers, inactive customers, and average credit limit
- Sidebar filters for age, inactivity, gender, income category, card category, and attrition status
- Dataset summary and model insight sections
- Interactive Plotly charts covering:
  - churn distribution
  - age and gender patterns
  - transaction behavior
  - inactivity behavior
  - credit usage and limit relationships
  - income, education, and segmentation views
- Floating Ask AI assistant anchored to the right side of the dashboard

## AI Assistant Capabilities

The assistant is designed to answer questions about:

- churn drivers
- risky customer groups
- current filtered dashboard view
- chart interpretation
- model insights and business recommendations

If Gemini is unavailable, the app falls back to a local dashboard-aware response path so the interface remains usable.

## Repository Structure

```text
.
├── BankChurners_Cleaned.csv
├── churn_insights.json
├── requirements.txt
├── README.md
├── tests/
│   └── test_dashboard_context.py
├── docs/
│   └── qa_workflow.md
└── web_app/
    ├── dashboard_app.py
    ├── llm_assistant.py
    ├── prompt_template.py
    └── analytics_context.py
```

## Local Setup

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Create a local `.env`

Use `.env.example` as the template.

```env
GEMINI_API_KEY=your_key_here
POWER_BI_REPORT_URL=https://app.powerbi.com/groups/me/reports/142efc4a-bb5c-43b5-8d54-297b41ab74d6/dc6b1f534010170e5201?experience=power-bi
POWER_BI_EMBED_URL=
```

Notes:

- `.env` is for local development only
- do not commit real credentials
- `POWER_BI_EMBED_URL` is optional and only needed if you have an actual embed URL

### 4. Run the app

```powershell
streamlit run web_app/dashboard_app.py
```

## Streamlit Community Cloud Deployment

Deploy this project as a public Streamlit Community Cloud app using the GitHub repository.

### Deployment settings

- Branch: `master`
- Main file path: `web_app/dashboard_app.py`

### Required Streamlit Cloud secrets

Add these in the Streamlit Cloud app settings under **Secrets**:

```toml
GEMINI_API_KEY = "your_key_here"
POWER_BI_REPORT_URL = "https://app.powerbi.com/groups/me/reports/142efc4a-bb5c-43b5-8d54-297b41ab74d6/dc6b1f534010170e5201?experience=power-bi"
POWER_BI_EMBED_URL = ""
```

Secret loading priority in the app is:

1. Streamlit secrets
2. Environment variables
3. Built-in Power BI report fallback

### Deployment notes

- Streamlit Cloud uses its own Secrets manager, not your local `.env`
- the app can still run without Gemini configured because it includes a fallback assistant path
- if `POWER_BI_EMBED_URL` is empty, the app will show the Power BI report as a secure external link
- full in-app Power BI embedding will only work once an actual embed URL is available

## Validation Checklist

Before deployment:

- confirm the app starts with `streamlit run web_app/dashboard_app.py`
- confirm charts, filters, KPIs, and the Ask AI panel load correctly
- confirm the app still works without a Gemini key
- confirm Gemini answers questions when a valid key is configured

Suggested acceptance questions:

- Which income category has the highest customer count?
- What does the credit usage scatter plot explain?
- Which customer segment looks most at risk?
- Ask a non-project question and confirm the assistant politely redirects

## Important Files

- `web_app/dashboard_app.py`: main Streamlit dashboard
- `web_app/llm_assistant.py`: Gemini integration and fallback response logic
- `web_app/prompt_template.py`: assistant prompt construction
- `web_app/analytics_context.py`: churn behavior summary helpers
- `churn_insights.json`: structured model insight payload
- `BankChurners_Cleaned.csv`: cleaned dataset used by the app

## Current Power BI Status

The project currently supports a Power BI report link fallback because publish-to-web embed creation is restricted. Once Power BI embed code creation is enabled by an administrator, `POWER_BI_EMBED_URL` can be supplied and the dashboard can render the embedded report directly inside the app.
