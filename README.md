# AI-Powered Customer Churn Analytics Dashboard

## Project Overview

This project combines machine learning, Power BI, and a large language model to create an end-to-end churn analytics system. Users can explore customer churn patterns through a dashboard and ask natural-language questions to an AI assistant that explains the most important drivers in business terms.

## End-to-End Architecture

Customer churn dataset  
-> data cleaning  
-> exploratory data analysis  
-> churn prediction models  
-> feature importance extraction  
-> structured insights JSON  
-> Power BI dashboard  
-> Streamlit web application  
-> LLM insight assistant

## Completed Solution Components

### Data and Modeling

- Cleaned the BankChurners dataset and removed unnecessary identifiers
- Encoded `Attrition_Flag` for modeling
- Completed exploratory data analysis on churn distribution, demographics, transactions, inactivity, and utilization
- Trained Logistic Regression and Random Forest models
- Selected Random Forest as the stronger model and extracted top churn drivers
- Generated `churn_insights.json` with dataset summary and model insight outputs

### Visualization Layer

- Built a six-page Power BI dashboard covering:
  - Customer Overview
  - Demographics
  - Card Category Analysis
  - Transaction Behavior
  - Credit Usage
  - Customer Engagement

### Web and LLM Layer

- Built a Streamlit dashboard interface in `web_app/dashboard_app.py`
- Loaded the cleaned dataset and insights JSON into the app
- Displayed churn KPIs, top churn drivers, and dashboard context
- Added a secured Power BI report link fallback when public embed is not available
- Implemented an LLM assistant module with reusable prompt templates
- Connected OpenAI response generation to the chatbot UI
- Added dashboard page focus selection for more targeted answers
- Enriched assistant context with churn behavior summaries computed from the dataset
- Added one-click example questions for faster user interaction
- Added smoke tests and a Q/A workflow checklist

## Key Files

- `web_app/dashboard_app.py`: main Streamlit app
- `web_app/llm_assistant.py`: OpenAI integration and response generation
- `web_app/prompt_template.py`: system and user prompt construction
- `web_app/analytics_context.py`: churn behavior summary builder
- `churn_insights.json`: structured ML insight payload
- `docs/qa_workflow.md`: manual validation checklist
- `tests/test_dashboard_context.py`: prompt/context smoke tests
- `.env.example`: local environment template

## Local Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Create or update the local `.env` file:

   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-4.1-mini
   POWER_BI_REPORT_URL=https://app.powerbi.com/groups/me/reports/142efc4a-bb5c-43b5-8d54-297b41ab74d6/dc6b1f534010170e5201?experience=power-bi
   POWER_BI_EMBED_URL=
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run web_app/dashboard_app.py
   ```

## Power BI Embedding Note

The current report URL is a secured Power BI report link, not a publish-to-web embed code. Because Power BI public embed creation is disabled by the tenant admin, the app currently shows a "View report" link instead of rendering the dashboard inside an iframe.

If embed creation is enabled later, set `POWER_BI_EMBED_URL` in `.env` to display the dashboard inside the web application.

## Example Questions

- Why are customers churning according to this dashboard?
- Which customer behaviors are the strongest warning signs?
- What does the Customer Engagement page imply about churn risk?
- Which card category looks most at risk?
- What actions should the business prioritize first?

## Testing

- Smoke tests: `tests/test_dashboard_context.py`
- Manual validation checklist: `docs/qa_workflow.md`

Note: test execution was not completed in this shell session because Python commands were hanging in the current environment.
