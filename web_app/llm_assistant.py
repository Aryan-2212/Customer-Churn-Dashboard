import os
from typing import Any

from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError
import streamlit as st
import google.generativeai as genai
from streamlit.errors import StreamlitSecretNotFoundError

from prompt_template import build_system_prompt, build_user_prompt


DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
load_dotenv()


def get_streamlit_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default))
    except StreamlitSecretNotFoundError:
        return default


def get_gemini_api_key() -> str:
    return get_streamlit_secret("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY", "")


def get_gemini_model() -> str:
    return get_streamlit_secret("GEMINI_MODEL") or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)


def get_gemini_debug_status() -> dict[str, bool]:
    streamlit_secret = get_streamlit_secret("GEMINI_API_KEY")
    env_var = os.getenv("GEMINI_API_KEY", "")
    return {
        "streamlit_secret_found": bool(streamlit_secret),
        "environment_variable_found": bool(env_var),
        "gemini_package_available": is_gemini_available(),
    }


def is_gemini_available() -> bool:
    try:
        import google.generativeai  # noqa: F401

        return True
    except ImportError:
        return False


def polish_response_text(response_text: str) -> str:
    cleaned = response_text.strip()
    replacements = {
        "The provided dashboard context": "From the dashboard",
        "the provided dashboard context": "the dashboard",
        "The current context": "The current view",
        "the current context": "the current view",
        "the dashboard context indicates": "the dashboard shows",
        "The dashboard context indicates": "The dashboard shows",
        "the current context does not contain": "this view does not show",
        "The current context does not contain": "This view does not show",
        "However, the dashboard indicates": "Still, the dashboard shows",
        "However, the provided context": "Still,",
        "I can see that ": "",
        "I can see ": "",
        "While the exact details shown in that specific scatter plot aren't available in this summary, ": "",
        "While the exact details aren't available in this summary, ": "",
        "While the exact details are not available in this summary, ": "",
        "The \"Credit Usage\" page of the dashboard focuses on ": "The chart is about ",
        "the \"Credit Usage\" page of the dashboard focuses on ": "the chart is about ",
    }
    for old_text, new_text in replacements.items():
        cleaned = cleaned.replace(old_text, new_text)
    cleaned = cleaned.replace("This suggests that", "That suggests")
    cleaned = cleaned.replace("This means that", "That means")
    return cleaned


def generate_fallback_response(question: str, context_payload: dict[str, Any]) -> str:
    summary = context_payload.get("dataset_summary", {})
    insights = context_payload.get("model_insights", {})
    patterns = context_payload.get("behavioral_patterns", {})
    segment = context_payload.get("segment_risk_snapshot", {})
    current_filters = context_payload.get("current_filters", {})
    snapshot = context_payload.get("dashboard_snapshot", {})
    chart_summaries = snapshot.get("chart_summaries", {})
    question_lower = question.lower()

    gratitude_tokens = ["thanks", "thank you", "ok", "okay", "great", "nice", "helpful"]
    if any(token == question_lower.strip() or question_lower.strip().startswith(f"{token} ") for token in gratitude_tokens):
        return "Happy to help. Ask me anything about the dashboard, churn drivers, risky segments, or the charts."

    project_tokens = [
        "churn",
        "customer",
        "dashboard",
        "chart",
        "graph",
        "plot",
        "income",
        "gender",
        "card",
        "credit",
        "limit",
        "utilization",
        "revolving",
        "transaction",
        "inactive",
        "inactivity",
        "education",
        "segment",
        "risk",
        "model",
    ]
    if not any(token in question_lower for token in project_tokens):
        return (
            "I can help with questions about this churn dashboard, the model insights, customer segments, or the charts. "
            "If you want, ask about churn drivers, risky groups, or what a specific visual means."
        )

    income_counts = snapshot.get("income_category_counts", {})
    gender_counts = snapshot.get("gender_counts", {})
    card_counts = snapshot.get("card_category_counts", {})
    education_counts = snapshot.get("education_level_counts", {})

    if income_counts and "income" in question_lower and any(
        token in question_lower for token in ["most", "highest", "count", "maximum", "top"]
    ):
        top_income = max(income_counts.items(), key=lambda item: item[1])
        return (
            f"In the current filtered view, `{top_income[0]}` has the highest customer count "
            f"with {top_income[1]:,} customers."
        )

    if card_counts and "card" in question_lower and any(
        token in question_lower for token in ["most", "highest", "count", "maximum", "top"]
    ):
        top_card = max(card_counts.items(), key=lambda item: item[1])
        return (
            f"In the current filtered view, `{top_card[0]}` has the highest customer count "
            f"with {top_card[1]:,} customers."
        )

    if gender_counts and "gender" in question_lower and any(
        token in question_lower for token in ["most", "highest", "count", "maximum", "top"]
    ):
        top_gender = max(gender_counts.items(), key=lambda item: item[1])
        return (
            f"In the current filtered view, `{top_gender[0]}` has the highest customer count "
            f"with {top_gender[1]:,} customers."
        )

    if education_counts and "education" in question_lower and any(
        token in question_lower for token in ["most", "highest", "count", "maximum", "top"]
    ):
        top_education = max(education_counts.items(), key=lambda item: item[1])
        return (
            f"In the current filtered view, `{top_education[0]}` has the highest customer count "
            f"with {top_education[1]:,} customers."
        )

    if chart_summaries and any(
        token in question_lower for token in ["scatter", "plot", "graph", "chart"]
    ):
        if any(token in question_lower for token in ["credit", "usage", "limit", "revolving"]):
            chart_info = chart_summaries.get("credit_usage_vs_limit", {})
            return (
                f"That scatter plot is comparing credit limit on the x-axis with revolving balance on the y-axis. "
                f"{chart_info.get('summary', '')} {chart_info.get('active_customer_pattern', '')} "
                f"{chart_info.get('churned_customer_pattern', '')} {chart_info.get('business_takeaway', '')}"
            ).strip()
        if any(token in question_lower for token in ["transaction", "activity"]):
            chart_info = chart_summaries.get("transaction_activity_vs_churn", {})
            return (
                f"That chart is meant to show how customer activity separates retained and churned behavior. "
                f"{chart_info.get('summary', '')} {chart_info.get('business_takeaway', '')}"
            ).strip()
        if any(token in question_lower for token in ["transaction count", "inactive month", "inactive months", "inactivity"]):
            chart_info = chart_summaries.get("transaction_count_vs_inactive_months", {})
            return (
                f"That chart is showing how transaction activity changes as customer inactivity increases. "
                f"{chart_info.get('summary', '')} {chart_info.get('business_takeaway', '')}"
            ).strip()
        if any(token in question_lower for token in ["inactivity", "utilization"]):
            chart_info = chart_summaries.get("inactivity_vs_credit_utilization", {})
            return (
                f"That chart shows how utilization changes as inactivity increases. "
                f"{chart_info.get('summary', '')} {chart_info.get('business_takeaway', '')}"
            ).strip()

    top_drivers = insights.get("top_churn_drivers", [])
    drivers_text = ", ".join(top_drivers[:3]) if top_drivers else "transaction behavior and inactivity"
    transaction_drop = (
        f"{patterns.get('avg_transaction_count_retained', 0):.1f} to "
        f"{patterns.get('avg_transaction_count_churned', 0):.1f}"
    )
    inactivity_rise = (
        f"{patterns.get('avg_inactive_months_retained', 0):.1f} to "
        f"{patterns.get('avg_inactive_months_churned', 0):.1f}"
    )
    risk_category = segment.get("highest_risk_card_category", "the currently filtered card mix")
    risk_rate = segment.get("highest_risk_card_churn_rate_percent", 0)
    filter_parts = []
    if current_filters:
        filter_parts.append(
            f"age {current_filters.get('age_range', {}).get('min', 'n/a')} to {current_filters.get('age_range', {}).get('max', 'n/a')}"
        )
        filter_parts.append(
            f"inactive months {current_filters.get('inactive_months_range', {}).get('min', 'n/a')} to {current_filters.get('inactive_months_range', {}).get('max', 'n/a')}"
        )
        if current_filters.get("gender") and current_filters.get("gender") != "All":
            filter_parts.append(f"gender {current_filters.get('gender')}")
        if current_filters.get("income_category") and current_filters.get("income_category") != "All":
            filter_parts.append(f"income {current_filters.get('income_category')}")
        if current_filters.get("card_category") and current_filters.get("card_category") != "All":
            filter_parts.append(f"card {current_filters.get('card_category')}")
        if current_filters.get("attrition_status") and current_filters.get("attrition_status") != "All":
            filter_parts.append(f"status {current_filters.get('attrition_status')}")
    filters_text = ", ".join(filter_parts) if filter_parts else "the current dashboard filters"

    return (
        f"Based on the filtered dashboard view for {filters_text}, churn is {summary.get('churn_rate_percent', 0):.2f}% "
        f"across {summary.get('dataset_size', 0):,} customers. The strongest churn signals in this view "
        f"are {drivers_text}. Customer activity weakens noticeably, with average transaction count moving from "
        f"{transaction_drop}, while inactivity increases from {inactivity_rise}. The riskiest visible segment is "
        f"{risk_category} at about {risk_rate:.2f}% churn. A practical next step is to target customers showing "
        f"low transaction activity and rising inactivity with early engagement or retention offers."
    )


def generate_llm_response(
    *,
    question: str,
    chat_history: list[dict[str, str]],
    context_payload: dict[str, Any],
    api_key: str,
    model: str,
) -> str:
    try:
        import google.generativeai as genai
    except ImportError as exc:
        return generate_fallback_response(question, context_payload)

    genai.configure(api_key=api_key)
    assistant_prompt = (
        f"{build_system_prompt(context_payload)}\n\n"
        f"{build_user_prompt(question, chat_history)}"
    )
    configured_candidates = []
    for candidate in [
        model,
        os.getenv("GEMINI_MODEL", ""),
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
    ]:
        if candidate and candidate not in configured_candidates:
            configured_candidates.append(candidate)

    available_candidates = []
    try:
        for listed_model in genai.list_models():
            supported_methods = getattr(listed_model, "supported_generation_methods", []) or []
            if "generateContent" in supported_methods:
                model_name = getattr(listed_model, "name", "")
                if model_name:
                    available_candidates.append(model_name.replace("models/", ""))
    except Exception:
        available_candidates = []

    candidate_models = []
    for candidate in configured_candidates + available_candidates:
        if candidate and candidate not in candidate_models:
            candidate_models.append(candidate)

    for candidate_model in candidate_models:
        try:
            response = genai.GenerativeModel(candidate_model).generate_content(assistant_prompt)
            return polish_response_text((response.text or "").strip())
        except Exception:
            continue

    return generate_fallback_response(question, context_payload)
