
import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

from commodity_finance_assistance.config_loader import load_config


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """Create and cache the configured OpenAI client."""
    load_dotenv()

    config = load_config()
    providers = config.get("llm", {}).get("providers", {})
    provider_name = config.get("llm", {}).get(
        "default_provider", "openai"
    )

    if provider_name != "openai":
        raise ValueError(
            f"This client supports only 'openai', not '{provider_name}'."
        )

    provider = providers.get("openai", {})
    key_env = provider.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.getenv(key_env)

    if not api_key:
        raise RuntimeError(
            f"Missing API key. Set {key_env} in your environment or .env."
        )

    client_args = {
        "api_key": api_key,
        "timeout": provider.get("timeout", 60),
        "max_retries": provider.get("max_retries", 2),
    }

    if provider.get("base_url"):
        client_args["base_url"] = provider["base_url"]

    return OpenAI(**client_args)
