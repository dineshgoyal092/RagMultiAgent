"""
LLM client — one function `chat()` that works for OpenAI, Groq, and Ollama.

All three providers expose the same OpenAI-compatible REST API, so the
same openai.OpenAI SDK client works for all of them — just point it at
a different base_url. Switching providers = one .env change, no code change.
"""
from openai import OpenAI
from src import config


def _get_client() -> OpenAI:
    """Return an OpenAI SDK client configured for the active provider."""

    if config.LLM_PROVIDER == "groq":
        # Groq runs the same models but on faster inference hardware (free tier available)
        return OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=config.GROQ_API_KEY,
        )

    if config.LLM_PROVIDER == "ollama":
        # Ollama runs models locally; api_key is unused but required by the SDK
        return OpenAI(
            base_url=config.OLLAMA_BASE_URL,
            api_key="ollama",  # placeholder — Ollama doesn't check the key
        )

    # Default: standard OpenAI (gpt-4o-mini, gpt-4o, etc.)
    return OpenAI(api_key=config.OPENAI_API_KEY)


def _get_model() -> str:
    """Return the model name string for the currently active provider."""
    if config.LLM_PROVIDER == "groq":   return config.GROQ_MODEL
    if config.LLM_PROVIDER == "ollama": return config.OLLAMA_MODEL
    return config.OPENAI_MODEL


def chat(messages: list, temperature: float = 0.1, max_tokens: int = 600) -> str:
    """
    Send a chat request to the LLM and return the response as a plain string.

    Args:
        messages    : list of {"role": "system"/"user"/"assistant", "content": "..."}
        temperature : 0   = fully deterministic (used for routing + judging)
                      0.1 = near-deterministic with slight variation (used for answers)
        max_tokens  : caps response length — routing uses 5, answers use 600
    """
    resp = _get_client().chat.completions.create(
        model=_get_model(),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # The actual text is always in choices[0].message.content
    return resp.choices[0].message.content
