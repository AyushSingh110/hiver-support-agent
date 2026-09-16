from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request

from src import config


class LLMError(RuntimeError):
    """Raised when the model cannot be reached or returns an unusable response."""


def _cache_key(prompt: str, model: str, temperature: float, seed: int, format_json: bool) -> str:
    payload = json.dumps(
        {"prompt": prompt, "model": model, "temperature": temperature,
         "seed": seed, "format_json": format_json},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(key: str):
    return config.CACHE_DIR / f"{key}.json"


def _post(url: str, body: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as error:
        raise LLMError(
            f"Could not reach Ollama at {url}: {error}. Is the daemon running?"
        ) from error


def complete(
    prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    seed: int = config.RANDOM_SEED,
    format_json: bool = True,
    use_cache: bool = True,
) -> dict:
    """Single completion from the local Ollama daemon, cached on disk by input hash."""
    model = model or config.GENERATION_MODEL
    key = _cache_key(prompt, model, temperature, seed, format_json)
    path = _cache_path(key)

    if use_cache and path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        cached["from_cache"] = True
        return cached

    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "seed": seed},
    }
    if format_json:
        body["format"] = "json"

    started = time.perf_counter()
    payload = _post(config.OLLAMA_URL, body, config.OLLAMA_TIMEOUT_SECONDS)
    elapsed = time.perf_counter() - started

    text = payload.get("response", "")
    if not isinstance(text, str) or not text.strip():
        raise LLMError(f"{model} returned an empty response")

    result = {
        "text": text,
        "model": model,
        "temperature": temperature,
        "seed": seed,
        "latency_seconds": round(elapsed, 3),
        "cache_key": key,
        "from_cache": False,
    }
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def cache_size() -> int:
    if not config.CACHE_DIR.exists():
        return 0
    return len(list(config.CACHE_DIR.glob("*.json")))
