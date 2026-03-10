"""Provider registry and model-resolution helpers for llm_runner."""

from __future__ import annotations

import logging
import os

import httpx
from pydantic import BaseModel
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider

from llm_runner.contracts import ProviderInfo, ProvidersListResponse

logger = logging.getLogger(__name__)


class ProviderConfig(BaseModel):
    """Provider-specific model-resolution configuration."""

    env_var: str | None
    base_url: str
    output_mode: str = "prompted"

    @property
    def effective_base_url(self) -> str:
        """Return the base URL passed to provider adapters."""
        return self.base_url

    def get_models(self) -> list[str]:
        """Return live model IDs for this provider."""
        return []


class GroqProviderConfig(ProviderConfig):
    """Groq chat model catalog."""

    env_var: str | None = "GROQ_API_KEY"
    base_url: str = "https://api.groq.com/openai/v1"
    output_mode: str = "prompted"

    _exclude = ("whisper", "guard", "orpheus", "safeguard")

    def get_models(self) -> list[str]:
        key = os.environ.get(self.env_var or "", "")
        if not key:
            return []
        try:
            response = httpx.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {key}"},
                timeout=8.0,
            )
            response.raise_for_status()
            payload = response.json()
            return [
                model["id"]
                for model in payload.get("data", [])
                if not any(
                    excluded in model["id"].lower() for excluded in self._exclude
                )
            ]
        except Exception as exc:
            logger.error("Failed to fetch Groq models: %s", exc)
            return []


class OpenRouterProviderConfig(ProviderConfig):
    """OpenRouter free-model catalog."""

    env_var: str | None = "OPENROUTER_API_KEY"
    base_url: str = "https://openrouter.ai/api/v1"
    output_mode: str = "prompted"

    def get_models(self) -> list[str]:
        key = os.environ.get(self.env_var or "", "")
        if not key:
            return []
        try:
            response = httpx.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {key}"},
                timeout=10.0,
            )
            response.raise_for_status()
            payload = response.json()
            return [
                model["id"]
                for model in payload.get("data", [])
                if ":free" in model["id"]
            ]
        except Exception as exc:
            logger.error("Failed to fetch OpenRouter models: %s", exc)
            return []


class NvidiaProviderConfig(ProviderConfig):
    """NVIDIA NIM model catalog."""

    env_var: str | None = "NVIDIA_NIM_API_KEY"
    base_url: str = "https://integrate.api.nvidia.com/v1"
    output_mode: str = "prompted"

    def get_models(self) -> list[str]:
        key = os.environ.get(self.env_var or "", "")
        if not key:
            return []
        try:
            response = httpx.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {key}"},
                timeout=8.0,
            )
            response.raise_for_status()
            payload = response.json()
            return [model["id"] for model in payload.get("data", [])]
        except Exception as exc:
            logger.error("Failed to fetch NVIDIA models: %s", exc)
            return []


class CloudflareProviderConfig(ProviderConfig):
    """Cloudflare Workers AI configuration."""

    env_var: str | None = "CLOUDFLARE_API_KEY"
    account_id_env_var: str = "CLOUDFLARE_ACCOUNT_ID"
    base_url: str = "https://api.cloudflare.com/client/v4/accounts"
    output_mode: str = "prompted"

    @property
    def effective_base_url(self) -> str:
        """Return the account-specific Cloudflare base URL."""
        account_id = os.environ.get(self.account_id_env_var, "")
        if not account_id:
            raise ValueError(
                f"{self.account_id_env_var} not set (required for Cloudflare Workers AI)"
            )
        return f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"

    def get_models(self) -> list[str]:
        account_id = os.environ.get(self.account_id_env_var, "")
        api_key = os.environ.get(self.env_var or "", "")
        if not api_key or not account_id:
            return []
        try:
            models: list[str] = []
            page = 1
            while True:
                response = httpx.get(
                    f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/models/search",
                    params={"per_page": 100, "page": page},
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=10.0,
                )
                response.raise_for_status()
                payload = response.json()
                batch = payload.get("result", [])
                models.extend(
                    model["name"]
                    for model in batch
                    if model.get("task", {}).get("name") == "Text Generation"
                )
                info = payload.get("result_info", {})
                if len(models) >= info.get("total_count", 0) or not batch:
                    return models
                page += 1
        except Exception as exc:
            logger.error("Failed to fetch Cloudflare models: %s", exc)
            return []


class MistralProviderConfig(ProviderConfig):
    """Mistral chat model catalog."""

    env_var: str | None = "MISTRAL_API_KEY"
    base_url: str = "https://api.mistral.ai/v1"
    output_mode: str = "tool"

    def get_models(self) -> list[str]:
        key = os.environ.get(self.env_var or "", "")
        if not key:
            return []
        try:
            response = httpx.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {key}"},
                timeout=8.0,
            )
            response.raise_for_status()
            payload = response.json()
            return [
                model["id"]
                for model in payload.get("data", [])
                if "completion"
                in " ".join(
                    str(capability) for capability in model.get("capabilities", [])
                )
                or "chat" in model["id"]
            ]
        except Exception as exc:
            logger.error("Failed to fetch Mistral models: %s", exc)
            return []


class OllamaCloudProviderConfig(ProviderConfig):
    """Ollama Cloud model catalog."""

    env_var: str | None = "OLLAMA_API_KEY"
    base_url: str = "https://ollama.com/v1"
    output_mode: str = "prompted"

    def get_models(self) -> list[str]:
        key = os.environ.get(self.env_var or "", "")
        if not key:
            return []
        try:
            response = httpx.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {key}"},
                timeout=8.0,
            )
            response.raise_for_status()
            payload = response.json()
            return [model["id"] for model in payload.get("data", [])]
        except Exception as exc:
            logger.error("Failed to fetch Ollama Cloud models: %s", exc)
            return []


class OllamaLocalProviderConfig(ProviderConfig):
    """Local Ollama model catalog."""

    env_var: str | None = None
    base_url: str = "http://localhost:11434/v1"
    output_mode: str = "prompted"

    def get_models(self) -> list[str]:
        try:
            response = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
            response.raise_for_status()
            payload = response.json()
            return [
                model["name"]
                for model in payload.get("models", [])
                if model.get("name", "").endswith(":cloud")
            ]
        except httpx.ConnectError:
            return []
        except Exception as exc:
            logger.error("Failed to fetch local Ollama models: %s", exc)
            return []


PROVIDERS: dict[str, ProviderConfig] = {
    "groq": GroqProviderConfig(),
    "openrouter": OpenRouterProviderConfig(),
    "nvidia": NvidiaProviderConfig(),
    "mistral": MistralProviderConfig(),
    "cloudflare": CloudflareProviderConfig(),
    "ollama-cloud": OllamaCloudProviderConfig(),
    "ollama": OllamaLocalProviderConfig(),
}


def resolve(slug: str) -> tuple[ProviderConfig, str]:
    """Resolve a provider/model slug into config and provider-local model ID."""
    if "/" not in slug:
        raise ValueError(
            f"Invalid model format {slug!r}. Expected provider/model "
            f"(known providers: {', '.join(PROVIDERS)})"
        )
    prefix, model_id = slug.split("/", 1)
    if prefix not in PROVIDERS:
        raise ValueError(
            f"Unknown provider {prefix!r}. Supported: {', '.join(PROVIDERS)}"
        )
    return PROVIDERS[prefix], model_id


def api_key(cfg: ProviderConfig) -> str:
    """Return the configured API key for a provider, if any."""
    if cfg.env_var is None:
        return "ollama"
    return os.environ.get(cfg.env_var) or ""


def make_model(
    cfg: ProviderConfig,
    model_id: str,
    slug: str,
) -> GroqModel | OpenAIChatModel:
    """Build the pydantic-ai model object for a provider/model slug."""
    key = api_key(cfg)
    profile = OpenAIModelProfile(default_structured_output_mode=cfg.output_mode)  # type: ignore[arg-type]
    if slug.startswith("groq/"):
        return GroqModel(model_id, provider=GroqProvider(api_key=key))
    if slug.startswith("openrouter/"):
        return OpenAIChatModel(
            model_id,
            provider=OpenRouterProvider(api_key=key),
            profile=profile,
        )
    return OpenAIChatModel(
        model_id,
        provider=OpenAIProvider(base_url=cfg.effective_base_url, api_key=key),
        profile=profile,
    )


def list_models(provider: str | None = None) -> list[str]:
    """Return provider/model slugs, optionally limited to one provider."""
    if provider is not None:
        cfg = PROVIDERS.get(provider)
        if cfg is None:
            raise ValueError(f"Unknown provider {provider!r}. Known: {list(PROVIDERS)}")
        return [f"{provider}/{model}" for model in cfg.get_models()]
    return [
        f"{provider_name}/{model}"
        for provider_name, cfg in PROVIDERS.items()
        for model in cfg.get_models()
    ]


def validate(slug: str) -> None:
    """Validate that a provider/model slug is usable in the current environment."""
    cfg, model_id = resolve(slug)
    prefix = slug.split("/", 1)[0]
    key = api_key(cfg)
    if not key and cfg.env_var is not None:
        raise ValueError(f"{cfg.env_var} not set (required for {slug})")
    if isinstance(cfg, CloudflareProviderConfig) and not os.environ.get(
        cfg.account_id_env_var
    ):
        raise ValueError(f"{cfg.account_id_env_var} not set (required for cloudflare/)")
    available = cfg.get_models()
    if available and model_id not in available:
        sample = available[:20]
        lines = "\n  ".join(f"{prefix}/{model}" for model in sample)
        extra = f"\n  ... and {len(available) - 20} more" if len(available) > 20 else ""
        raise ValueError(f"Model {slug!r} not found. Available:\n  {lines}{extra}")


def list_providers_response() -> ProvidersListResponse:
    """Build the JSON response for `providers list`."""
    return ProvidersListResponse(
        providers=[
            ProviderInfo(name=name, models=list_models(name))
            for name in sorted(PROVIDERS)
        ]
    )


__all__ = [
    "PROVIDERS",
    "ProviderConfig",
    "api_key",
    "list_models",
    "list_providers_response",
    "make_model",
    "resolve",
    "validate",
]
