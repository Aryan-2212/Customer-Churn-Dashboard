import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from analytics_context import build_behavioral_summary
from llm_assistant import generate_llm_response, get_openai_api_key, get_openai_model
from prompt_template import build_dashboard_context_payload


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DATA_PATH = PROJECT_ROOT / "BankChurners_Cleaned.csv"
INSIGHTS_PATH = PROJECT_ROOT / "churn_insights.json"
DEFAULT_POWER_BI_REPORT_URL = (
    "https://app.powerbi.com/groups/me/reports/"
    "142efc4a-bb5c-43b5-8d54-297b41ab74d6/dc6b1f534010170e5201?experience=power-bi"
)

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


def get_power_bi_links() -> tuple[str, str]:
    power_bi_config = st.secrets.get("power_bi", {})
    embed_url = power_bi_config.get("embed_url", "") or os.getenv("POWER_BI_EMBED_URL", "")
    report_url = power_bi_config.get("report_url", "") or os.getenv(
        "POWER_BI_REPORT_URL", DEFAULT_POWER_BI_REPORT_URL
    )
    return embed_url, report_url


def initialize_chat_state() -> None:
    if "assistant_messages" not in st.session_state:
        st.session_state.assistant_messages = [
            {
                "role": "assistant",
                "content": (
                    "Ask about churn drivers, risky customer behavior, or what each dashboard page means."
                ),
            }
        ]


def render_assistant_panel(
    context_payload: dict, example_queries: list[str], dashboard_pages: list[tuple[str, str]]
) -> None:
    st.subheader("LLM Insight Assistant")
    st.write(
        "This assistant combines the dashboard context and ML churn drivers to explain what the report means in business terms."
    )

    page_options = ["All Dashboard Pages"] + [page_name for page_name, _ in dashboard_pages]
    selected_page = st.selectbox("Assistant focus", page_options)
    selected_page_context = next(
        (
            {"page": page_name, "description": description}
            for page_name, description in dashboard_pages
            if page_name == selected_page
        ),
        None,
    )
    active_context_payload = {
        **context_payload,
        "assistant_focus": selected_page_context or {
            "page": "All Dashboard Pages",
            "description": "Use the full dashboard context when answering.",
        },
    }

    api_key = get_openai_api_key(st.secrets)
    model_name = get_openai_model(st.secrets)
    if api_key:
        st.caption(f"OpenAI model configured: `{model_name}`")
    else:
        st.warning(
            "Add `OPENAI_API_KEY` or `st.secrets['openai']['api_key']` to enable live responses."
        )

    st.markdown("**Suggested questions**")
    for question in example_queries:
        st.markdown(f"- {question}")

    for message in st.session_state.assistant_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Ask the churn assistant a question")
    if not question:
        return

    st.session_state.assistant_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        if not api_key:
            fallback_response = (
                "The assistant UI is ready, but no OpenAI API key is configured yet. "
                "Add the key and try again to generate live churn explanations."
            )
            st.write(fallback_response)
            st.session_state.assistant_messages.append(
                {"role": "assistant", "content": fallback_response}
            )
            return

        with st.spinner("Generating insight..."):
            try:
                answer = generate_llm_response(
                    question=question,
                    chat_history=st.session_state.assistant_messages[:-1],
                    context_payload=active_context_payload,
                    api_key=api_key,
                    model=model_name,
                )
            except Exception as exc:
                answer = (
                    "I couldn't generate a response right now. "
                    f"Please verify the OpenAI package, API key, and model configuration. Details: {exc}"
                )

        st.write(answer)
        st.session_state.assistant_messages.append({"role": "assistant", "content": answer})


df = load_data()
insights = load_insights()
embed_url, report_url = get_power_bi_links()
churned_customers = int(df["Attrition_Flag"].sum()) if "Attrition_Flag" in df.columns else 0
top_drivers = insights.get("top_churn_drivers", [])
context_payload = build_dashboard_context_payload(
    dataset_size=len(df),
    churn_rate=float(insights.get("churn_rate", 0)),
    churned_customers=churned_customers,
    best_model=insights.get("best_model", "Not available"),
    top_drivers=top_drivers,
    dashboard_pages=DASHBOARD_PAGES,
)
context_payload.update(build_behavioral_summary(df))
example_queries = [
    "Why are customers churning according to this dashboard?",
    "Which customer behaviors are the strongest warning signs?",
    "What actions should the business prioritize to reduce churn?",
]

initialize_chat_state()

st.title("AI-Powered Customer Churn Analytics Dashboard")
st.caption(
    "Explore customer churn performance, embedded business intelligence, and AI-generated explanations in one interface."
)

summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
summary_col1.metric("Total Customers", f"{len(df):,}")
summary_col2.metric("Churned Customers", f"{churned_customers:,}")
summary_col3.metric("Churn Rate", f"{float(insights.get('churn_rate', 0)):.2f}%")
summary_col4.metric("Best Model", insights.get("best_model", "Not available"))

overview_col, assistant_col = st.columns([1.2, 0.8], gap="large")

with overview_col:
    st.subheader("Dataset Summary")
    st.write(
        "The cleaned churn dataset is loaded into the app and connected to the feature-importance insights used by the assistant."
    )
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Model Insights")
    st.write("Top churn drivers extracted from the Random Forest model:")
    for index, driver in enumerate(top_drivers, start=1):
        st.markdown(f"{index}. `{driver}`")

    st.subheader("Dashboard Context")
    for page_name, description in DASHBOARD_PAGES:
        st.markdown(f"**{page_name}**: {description}")

with assistant_col:
    render_assistant_panel(context_payload, example_queries, DASHBOARD_PAGES)

st.subheader("Power BI Experience")
if embed_url:
    components.html(
        f"""
        <iframe
            title="Customer Churn Power BI Dashboard"
            width="100%"
            height="720"
            src="{embed_url}"
            frameborder="0"
            allowFullScreen="true">
        </iframe>
        """,
        height=740,
    )
else:
    st.info(
        "Public embed code is not enabled for this report, so the dashboard cannot be rendered in an iframe yet."
    )
    st.markdown(
        f"Open the secured Power BI report here: [View report]({report_url})"
    )
    st.caption(
        "Once your admin enables embed code creation, add the publish-to-web or secure embed URL to enable in-app iframe rendering."
    )
