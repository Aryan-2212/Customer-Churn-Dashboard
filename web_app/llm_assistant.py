import os
from typing import Any

from dotenv import load_dotenv

from prompt_template import build_system_prompt, build_user_prompt


DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
load_dotenv()


def get_gemini_api_key() -> str:
    return os.getenv("GEMINI_API_KEY", "")


def get_gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)


def is_gemini_available() -> bool:
    try:
        import google.generativeai  # noqa: F401

        return True
    except ImportError:
        return False


def generate_fallback_response(question: str, context_payload: dict[str, Any]) -> str:
    summary = context_payload.get("dataset_summary", {})
    insights = context_payload.get("model_insights", {})
    patterns = context_payload.get("behavioral_patterns", {})
    segment = context_payload.get("segment_risk_snapshot", {})

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

    return (
        f"Based on the filtered dashboard view, churn is {summary.get('churn_rate_percent', 0):.2f}% "
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
    candidate_models = []
    for candidate in [
        model,
        os.getenv("GEMINI_MODEL", ""),
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
    ]:
        if candidate and candidate not in candidate_models:
            candidate_models.append(candidate)

    last_error = None
    for candidate_model in candidate_models:
        try:
            response = genai.GenerativeModel(candidate_model).generate_content(assistant_prompt)
            return (response.text or "").strip()
        except Exception as exc:
            last_error = exc
            continue

    return generate_fallback_response(question, context_payload)
