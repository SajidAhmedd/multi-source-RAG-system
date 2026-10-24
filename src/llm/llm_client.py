"""
LLM client with automatic provider fallback.

WHY THIS FILE EXISTS:
- Free LLM tiers are rate-limited; if one provider rejects or fails a request,
  the same request is retried on the next provider instead of failing
- Groq, Google Gemini and Mistral all expose OpenAI-compatible APIs, so one
  client (the openai package) talks to all of them
- Mirrors the Groq/OpenAI interface (client.chat.completions.create), so
  calling code doesn't change

CONFIGURATION (environment variables):
- LLM_PROVIDERS: comma-separated order to try, e.g. "mistral,groq"
- MISTRAL_API_KEY / GROQ_API_KEY / GEMINI_API_KEY (or GOOGLE_API_KEY)
- MISTRAL_MODEL / GROQ_MODEL / GEMINI_MODEL to override the default models
"""

import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional

from openai import OpenAI


@dataclass
class Provider:
    name: str
    base_url: str
    api_key_vars: List[str]
    model_var: str
    default_model: str
    # Extra request parameters only this provider understands
    extra_body: Dict = field(default_factory=dict)

    @property
    def api_key(self) -> Optional[str]:
        for var in self.api_key_vars:
            value = os.getenv(var, "").strip()
            if value and not value.startswith("your_"):
                return value
        return None

    @property
    def model(self) -> str:
        return os.getenv(self.model_var) or self.default_model


PROVIDERS = {
    "groq": Provider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_vars=["GROQ_API_KEY"],
        model_var="GROQ_MODEL",
        default_model="openai/gpt-oss-20b",
        # gpt-oss models reason before answering; keep that short so max_tokens goes to the answer
        extra_body={"reasoning_effort": "low"},
    ),
    "gemini": Provider(
        name="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_vars=["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        model_var="GEMINI_MODEL",
        default_model="gemini-flash-latest",
    ),
    "mistral": Provider(
        name="mistral",
        base_url="https://api.mistral.ai/v1",
        api_key_vars=["MISTRAL_API_KEY"],
        model_var="MISTRAL_MODEL",
        default_model="mistral-small-latest",
    ),
}


class _Completions:
    def __init__(self, client: "LLMClient"):
        self._client = client

    def create(self, messages: list, model: Optional[str] = None, **kwargs):
        """
        Send a chat completion, trying each configured provider in order.

        `model` is accepted for compatibility and ignored: every provider
        uses its own configured model.
        """
        errors = []
        for provider, client in self._client.providers:
            try:
                response = client.chat.completions.create(
                    model=provider.model,
                    messages=messages,
                    extra_body=provider.extra_body or None,
                    **kwargs,
                )
                if response.choices and response.choices[0].message.content:
                    self._client.last_provider = provider.name
                    return response
                errors.append(f"{provider.name}: empty response")
            except Exception as e:
                errors.append(f"{provider.name}: {str(e)[:200]}")
                print(f"LLM provider {provider.name} failed, trying next: {str(e)[:120]}")

        raise RuntimeError("All LLM providers failed - " + " | ".join(errors))


class LLMClient:
    """
    Drop-in replacement for groq.Groq() / openai.OpenAI() chat completions.

    Usage:
        client = LLMClient()
        response = client.chat.completions.create(messages=[...], temperature=0.3)
        print(response.choices[0].message.content)
    """

    def __init__(self, order: Optional[str] = None):
        order = order or os.getenv("LLM_PROVIDERS", "groq")
        self.providers = []
        for name in [n.strip().lower() for n in order.split(",") if n.strip()]:
            provider = PROVIDERS.get(name)
            if provider is None:
                raise ValueError(f"Unknown LLM provider '{name}'. Use: {', '.join(PROVIDERS)}")
            if provider.api_key:
                # Fail over quickly instead of waiting on long retry loops
                client = OpenAI(api_key=provider.api_key, base_url=provider.base_url, max_retries=1, timeout=90)
                self.providers.append((provider, client))

        if not self.providers:
            raise ValueError(
                f"No API key set for any provider in LLM_PROVIDERS={order}. "
                "Set at least one of MISTRAL_API_KEY, GROQ_API_KEY or GEMINI_API_KEY."
            )

        self.last_provider: Optional[str] = None
        self.chat = SimpleNamespace(completions=_Completions(self))

    @property
    def model(self) -> str:
        """Model of the primary provider (for display)."""
        return self.providers[0][0].model

    @property
    def provider_names(self) -> List[str]:
        return [p.name for p, _ in self.providers]


def configured_providers(order: Optional[str] = None) -> List[str]:
    """Names of providers in LLM_PROVIDERS that have an API key set."""
    order = order or os.getenv("LLM_PROVIDERS", "groq")
    return [
        n.strip().lower() for n in order.split(",")
        if n.strip().lower() in PROVIDERS and PROVIDERS[n.strip().lower()].api_key
    ]
