import os

from dotenv import load_dotenv

from prompt_template import build_system_prompt, build_user_prompt


DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"
load_dotenv()


def get_gemini_api_key() -> str:
    return os.getenv("GEMINI_API_KEY", "")


def get_gemini_model() -> str:
    return DEFAULT_GEMINI_MODEL


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
        raise RuntimeError(
            "The `google-generativeai` package is not installed. Install it before using the assistant."
        ) from exc

    genai.configure(api_key=api_key)
    assistant_prompt = (
        f"{build_system_prompt(context_payload)}\n\n"
        f"{build_user_prompt(question, chat_history)}"
    )
    response = genai.GenerativeModel(model).generate_content(assistant_prompt)
    return (response.text or "").strip()
