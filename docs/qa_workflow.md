# Question-Answering Workflow Test Plan

## Goal

Validate that the Streamlit dashboard and LLM assistant work together for churn insight explanations.

## Preconditions

- Install dependencies with `pip install -r requirements.txt`
- Add the OpenAI key to the local `.env` file
- Start the app with `streamlit run web_app/dashboard_app.py`

## Functional Checks

1. Confirm the dashboard summary loads and shows total customers, churned customers, churn rate, and best model.
2. Confirm top churn drivers appear in the model insights section.
3. Confirm the dashboard context lists all six Power BI pages.
4. Confirm the secured Power BI report link is visible when no embed URL is configured.
5. Confirm the assistant focus selector changes between all pages and specific page contexts.
6. Confirm one-click example queries appear and send a prompt into the chat flow.
7. Confirm manual chat questions can be submitted from the chat input.
8. Confirm the assistant warns clearly when the OpenAI API key is missing.
9. Confirm the assistant returns a business-style answer when the API key is configured.
10. Confirm the response stays grounded in churn drivers and does not invent unavailable dashboard metrics.

## Suggested Test Questions

- Why are customers churning according to this dashboard?
- Which customer behaviors are the strongest warning signs?
- What does the Customer Engagement page imply about churn risk?
- Which card category looks most at risk?
- What actions should the business prioritize first?

## Expected Quality Bar

- Answers should mention relevant churn drivers when appropriate.
- Answers should reference inactivity, transactions, utilization, or segment risk when supported by context.
- Answers should acknowledge missing evidence instead of guessing.
- Answers should end with a practical recommendation when the question asks for action.
