import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DATA_PATH = PROJECT_ROOT / "BankChurners_Cleaned.csv"
INSIGHTS_PATH = PROJECT_ROOT / "churn_insights.json"

DASHBOARD_PAGES = [
    (
        "Customer Overview",
        "High-level KPIs for total customers, churn rate, active vs attrited segments, and overall account health.",
    ),
    (
        "Demographics",
        "Breakdown of churn trends by age band, gender, education level, marital status, and income category.",
    ),
    (
        "Card Category Analysis",
        "Comparison of churn behavior across card tiers to identify higher-risk product segments.",
    ),
    (
        "Transaction Behavior",
        "Analysis of transaction amount, transaction count, and changing spend patterns linked to churn risk.",
    ),
    (
        "Credit Usage",
        "Focus on revolving balance, utilization ratio, credit limit, and repayment behavior.",
    ),
    (
        "Customer Engagement",
        "Tracks inactivity, relationship duration, contact frequency, and engagement signals that precede churn.",
    ),
]


st.set_page_config(page_title="AI-Powered Customer Churn Dashboard", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_insights() -> dict:
    with INSIGHTS_PATH.open(encoding="utf-8") as insights_file:
        return json.load(insights_file)


def get_power_bi_embed_url() -> str:
    power_bi_config = st.secrets.get("power_bi", {})
    secret_url = power_bi_config.get("embed_url", "")
    return secret_url or os.getenv("POWER_BI_EMBED_URL", "")


df = load_data()
insights = load_insights()
power_bi_url = get_power_bi_embed_url()
churned_customers = int(df["Attrition_Flag"].sum()) if "Attrition_Flag" in df.columns else 0


st.title("AI-Powered Customer Churn Analytics Dashboard")
st.caption(
    "Explore customer churn performance, model-driven insights, and dashboard context in one interface."
)

summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
summary_col1.metric("Total Customers", f"{len(df):,}")
summary_col2.metric("Churned Customers", f"{churned_customers:,}")
summary_col3.metric("Churn Rate", f"{insights.get('churn_rate', 0):.2f}%")
summary_col4.metric("Best Model", insights.get("best_model", "Not available"))

left_col, right_col = st.columns([1.3, 1], gap="large")

with left_col:
    st.subheader("Dataset Summary")
    st.write(
        "The cleaned churn dataset is loaded into the app and ready for interactive exploration."
    )
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Model Insights")
    st.write(
        "Key churn signals extracted from the machine learning pipeline and prepared for the LLM assistant."
    )
    drivers = insights.get("top_churn_drivers", [])
    for index, driver in enumerate(drivers, start=1):
        st.markdown(f"{index}. `{driver}`")

with right_col:
    st.subheader("Dashboard Context")
    st.write(
        "The Power BI report is organized into six business views so the assistant can explain each section with the right context."
    )
    for page_name, description in DASHBOARD_PAGES:
        st.markdown(f"**{page_name}**: {description}")

st.subheader("Embedded Power BI Dashboard")
if power_bi_url:
    components.html(
        f"""
        <iframe
            title="Customer Churn Power BI Dashboard"
            width="100%"
            height="720"
            src="{power_bi_url}"
            frameborder="0"
            allowFullScreen="true">
        </iframe>
        """,
        height=740,
    )
else:
    st.info(
        "Add the Power BI publish URL in `st.secrets['power_bi']['embed_url']` or the "
        "`POWER_BI_EMBED_URL` environment variable to display the embedded dashboard."
    )

st.subheader("Assistant Preview")
st.write(
    "The LLM insight assistant will use the dataset summary, model insights, and dashboard page context shown above to answer natural-language questions in the next phase."
)
st.markdown(
    """
    Example questions the assistant will support:
    - Why are customers churning?
    - Which customer segment appears most at risk?
    - Which churn drivers should the business act on first?
    """
)
