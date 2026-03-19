import json
from typing import Any


def build_dashboard_context_payload(
    *,
    dataset_size: int,
    churn_rate: float,
    churned_customers: int,
    best_model: str,
    top_drivers: list[str],
    dashboard_pages: list[tuple[str, str]],
) -> dict[str, Any]:
    return {
        "dataset_summary": {
            "dataset_size": dataset_size,
            "churned_customers": churned_customers,
            "churn_rate_percent": round(churn_rate, 2),
            "best_model": best_model,
        },
        "model_insights": {
            "top_churn_drivers": top_drivers,
        },
        "dashboard_pages": [
            {"page": page_name, "description": description}
            for page_name, description in dashboard_pages
        ],
    }


def build_system_prompt(context_payload: dict[str, Any]) -> str:
    serialized_context = json.dumps(context_payload, indent=2)
    return f"""
You are an AI insight assistant for a customer churn analytics dashboard.

Your job:
- Explain churn behavior in clear business language.
- Use only the provided dashboard context, dataset summary, and model insights.
- Tie answers back to the known churn drivers whenever possible.
- Use the behavioral summary and risk snapshots when the question asks for patterns or risky segments.
- Be explicit when the dashboard context does not contain enough evidence.
- Avoid fabricating metrics, segments, or trends that are not in the context.

Response style:
- Keep answers concise but insightful.
- Prioritize business interpretation over technical jargon.
- When useful, end with a practical recommendation grounded in the supplied churn patterns.

Structured dashboard context:
{serialized_context}
""".strip()


def build_user_prompt(question: str, chat_history: list[dict[str, str]]) -> str:
    recent_history = chat_history[-6:]
    history_lines = []
    for message in recent_history:
        role = message.get("role", "user").upper()
        content = message.get("content", "").strip()
        history_lines.append(f"{role}: {content}")

    history_block = "\n".join(history_lines) if history_lines else "No previous conversation."
    return f"""
Conversation so far:
{history_block}

Latest user question:
{question.strip()}

Answer using the dashboard context above. If the question asks for unsupported detail, say what is known and what is missing.
""".strip()
