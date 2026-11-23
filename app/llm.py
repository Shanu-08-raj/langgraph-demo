"""LLM configuration and initialization for the workflow.

Provides a configured ChatOllama instance using settings loaded from
environment variables or default fallback values.
"""

import logging
import os
from typing import Optional
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llama3.2:1b"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TEMPERATURE = 0.3

# Module-level cached instance for reuse across nodes
_llm_instance: Optional[ChatOllama] = None


def get_model_name() -> str:
    """Return the configured Ollama model name."""
    return os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)


def get_base_url() -> str:
    """Return the configured Ollama base URL."""
    return os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL)


def get_llm(temperature: float = DEFAULT_TEMPERATURE) -> ChatOllama:
    """Return a reusable ChatOllama model instance.

    Uses a cached instance if temperature matches the default, avoiding
    unnecessary duplicate client allocations across graph nodes.

    Args:
        temperature: Sampling temperature for generation.

    Returns:
        ChatOllama: Configured LangChain Ollama chat model.
    """
    global _llm_instance

    model = get_model_name()
    base_url = get_base_url()

    if temperature == DEFAULT_TEMPERATURE and _llm_instance is not None:
        return _llm_instance

    try:
        instance = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
        )
        if temperature == DEFAULT_TEMPERATURE:
            _llm_instance = instance
        return instance
    except Exception as exc:
        logger.error(
            "Failed to initialize ChatOllama with model '%s' at '%s': %s",
            model,
            base_url,
            exc,
        )
        raise ConnectionError(
            f"Could not initialize Ollama model '{model}' at '{base_url}'. "
            "Please check that Ollama is installed and running."
        ) from exc


def check_ollama_availability() -> bool:
    """Check whether the configured Ollama server is reachable and model is available.

    Returns:
        bool: True if server is reachable, False otherwise.
    """
    import urllib.request
    import urllib.error

    base_url = get_base_url().rstrip("/")
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.warning("Ollama server check failed at %s: %s", base_url, exc)
        return False
