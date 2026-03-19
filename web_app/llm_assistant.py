import os
from typing import Any

from prompt_template import build_system_prompt, build_user_prompt


DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"


def get_openai_api_key(secrets: Any) -> str:
    openai_config = secrets.get("openai", {})
    secret_key = openai_config.get("api_key", "")
    return secret_key or os.getenv("OPENAI_API_KEY", "")


def get_openai_model(secrets: Any) -> str:
    openai_config = secrets.get("openai", {})
    secret_model = openai_config.get("model", "")
    return secret_model or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)


def generate_llm_response(
    *,
    question: str,
    chat_history: list[dict[str, str]],
    context_payload: dict[str, Any],
    api_key: str,
    model: str,
) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The `openai` package is not installed. Install it before using the assistant."
        ) from exc

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": build_system_prompt(context_payload)},
            {"role": "user", "content": build_user_prompt(question, chat_history)},
        ],
    )
    return (response.output_text or "").strip()
