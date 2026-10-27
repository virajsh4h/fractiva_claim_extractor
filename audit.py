"""Append-only audit log for every model call."""

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def append_llm_log(
    log_path: str,
    prompt: str,
    response_raw: str,
    model: str,
    latency_s: float,
    parameters: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    system_instruction: Optional[str] = None,
    request_parts: Optional[list] = None,
) -> None:
    """Append one request/response pair.

    The user prompt is logged verbatim and hashed. The system instruction is
    logged separately and hashed too, and non-text request parts are recorded as
    hashes so the exact call can be reconstructed without storing images.
    """

    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model": model,
        "latency_seconds": round(latency_s, 3),
        "parameters": parameters or {},
        "prompt_sha256": prompt_sha256(prompt),
        "prompt": prompt,
        "system_instruction": system_instruction,
        "system_instruction_sha256": (
            prompt_sha256(system_instruction) if system_instruction else None
        ),
        "request_parts": request_parts or [],
        "response_raw": response_raw,
        "metadata": metadata or {},
    }
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
