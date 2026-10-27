"""Runtime configuration.

The API key is read from the environment or a local ``.env`` file. It is never
written to logs, predictions or commits.
"""

import os
from typing import Optional

DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_NVIDIA_EXTRACTION_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
DEFAULT_NVIDIA_VISION_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_OPENROUTER_TEXT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
DEFAULT_OPENROUTER_VISION_MODEL = "nvidia/nemotron-3.5-super-vl:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT_MS = 120_000
NVIDIA_TIMEOUT_S = 120
OPENROUTER_TIMEOUT_S = 120


def load_local_env(path: str = ".env") -> None:
    """Load simple KEY=VALUE lines without overwriting the environment."""

    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        # A malformed or unreadable .env must not crash the verifier. The
        # missing-key path remains fail-closed.
        return


def get_api_key() -> Optional[str]:
    load_local_env()
    return os.environ.get("GEMINI_API_KEY") or None


def get_model_name(default: Optional[str] = None) -> str:
    load_local_env()
    return os.environ.get("GEMINI_MODEL", default or DEFAULT_MODEL)


def get_nvidia_api_key() -> Optional[str]:
    load_local_env()
    return os.environ.get("NVIDIA_API_KEY") or None


def get_nvidia_extraction_model(default: Optional[str] = None) -> str:
    load_local_env()
    return os.environ.get(
        "NVIDIA_EXTRACTION_MODEL",
        default or DEFAULT_NVIDIA_EXTRACTION_MODEL,
    )


def get_nvidia_vision_model(default: Optional[str] = None) -> str:
    load_local_env()
    return os.environ.get(
        "NVIDIA_VISION_MODEL",
        default or DEFAULT_NVIDIA_VISION_MODEL,
    )


def get_openrouter_api_key() -> Optional[str]:
    load_local_env()
    return (
        os.environ.get("OPENROUTER_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or None
    )


def get_openrouter_text_model(default: Optional[str] = None) -> str:
    load_local_env()
    return os.environ.get(
        "OPENROUTER_TEXT_MODEL",
        default or DEFAULT_OPENROUTER_TEXT_MODEL,
    )


def get_openrouter_vision_model(default: Optional[str] = None) -> str:
    load_local_env()
    return os.environ.get(
        "OPENROUTER_VISION_MODEL",
        default or DEFAULT_OPENROUTER_VISION_MODEL,
    )


def get_primary_provider() -> str:
    load_local_env()
    return os.environ.get("PRIMARY_PROVIDER", "gemini").lower()

