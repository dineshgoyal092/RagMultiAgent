"""
LLM client — single function `chat()` works across OpenAI, Groq, and Ollama.

All three use the OpenAI-compatible API, so the same client code works for all.
Switch provider by changing LLM_PROVIDER in .env — no code changes needed.
"""
from openai import OpenAI
from src import config


def _get_client() -> OpenAI:
    """Return an OpenAI client pointed at the right provider."""
    if config.LLM_PROVIDER == "groq":
        return OpenAI(base_url="https://api.groq.com/openai/v1", api_key=config.GROQ_API_KEY)
    if config.LLM_PROVIDER == "ollama":
        return OpenAI(base_url=config.OLLAMA_BASE_URL, api_key="ollama")
    return OpenAI(api_key=config.OPENAI_API_KEY)


def _get_model() -> str:
    """Return the model name for the active provider."""
    if config.LLM_PROVIDER == "groq":   return config.GROQ_MODEL
    if config.LLM_PROVIDER == "ollama": return config.OLLAMA_MODEL
    return config.OPENAI_MODEL


def chat(messages: list, temperature: float = 0.1, max_tokens: int = 600) -> str:
    """Send a chat request and return the response text."""
    resp = _get_client().chat.completions.create(
        model=_get_model(),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content
