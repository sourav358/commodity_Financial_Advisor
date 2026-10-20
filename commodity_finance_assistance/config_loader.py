from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yml"


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Load application configuration from config.yml."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError("config.yml must contain a YAML mapping.")

    return config


def get_model_config(role: str = "chat") -> dict[str, Any]:
    """Return configuration for a named model role."""
    config = load_config()
    models = config.get("llm", {}).get("models", {})

    if role not in models:
        raise KeyError(f"Unknown model role: {role}")

    return models[role]