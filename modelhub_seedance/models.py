"""Model catalogue: live ``GET /v1/models`` with a static fallback."""
from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional

from .client import DEFAULT_BASE_URL, ModelHubClient

_DUR_15 = [str(i) for i in range(4, 16)]
_DUR_30 = [str(i) for i in range(4, 31)]
_RATIOS = ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"]

# This pack only exposes the Seedance family (ids starting with this prefix).
MODEL_PREFIX = "seedance-"

# Snapshot of GET /v1/models (Seedance video models), 2026-11-26. Used when the
# catalogue cannot be fetched (offline start, network error).
FALLBACK_MODELS: List[Dict[str, Any]] = [
    {"id": "seedance-2.0", "ratio": _RATIOS + ["auto", "adaptive"], "resolution": ["480p", "720p", "1080p", "4k"],
     "duration": _DUR_15, "generate_audio": True},
    {"id": "seedance-2.0-fast", "ratio": _RATIOS + ["auto"], "resolution": ["480p", "720p", "1080p", "2k", "4k"],
     "duration": _DUR_15, "generate_audio": True},
    {"id": "seedance-2.0-mini", "ratio": _RATIOS + ["adaptive"], "resolution": ["480p", "720p", "1080p", "2k", "4k"],
     "duration": _DUR_15, "generate_audio": True},
    {"id": "seedance-2.5", "ratio": _RATIOS + ["auto"], "resolution": ["480p", "720p", "1080p", "4K"],
     "duration": _DUR_30, "generate_audio": True},
    {"id": "seedance-2.5-flash", "ratio": _RATIOS, "resolution": ["480p", "720p"],
     "duration": _DUR_30, "generate_audio": False},
]

_lock = threading.Lock()
_cache: Optional[List[Dict[str, Any]]] = None
_cache_source = "none"


def normalize_model(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Turn one ``/v1/models`` entry into ``{id, display_name, ratio, resolution, duration, generate_audio}``.
    Returns None for non-video and non-Seedance models."""
    if raw.get("media_type") != "video" or not str(raw.get("id") or "").startswith(MODEL_PREFIX):
        return None
    params = {p.get("name"): p for p in raw.get("parameters") or [] if isinstance(p, dict)}

    def values(name: str) -> List[str]:
        p = params.get(name) or {}
        return [str(v) for v in (p.get("accepted_values") or [])]

    def default(name: str) -> Optional[str]:
        p = params.get(name) or {}
        d = p.get("default")
        return None if d is None else str(d)

    return {
        "id": str(raw["id"]),
        "display_name": raw.get("display_name") or raw["id"],
        "ratio": values("ratio"),
        "resolution": values("resolution"),
        "duration": values("duration"),
        "generate_audio": "generate_audio" in params,
        "defaults": {k: default(k) for k in ("ratio", "resolution", "duration")},
    }


def fetch_models(base_url: str = DEFAULT_BASE_URL, timeout: float = 5.0, client: Optional[ModelHubClient] = None
                 ) -> List[Dict[str, Any]]:
    client = client or ModelHubClient(base_url=base_url)
    out = []
    for raw in client.list_models(timeout=timeout):
        m = normalize_model(raw) if isinstance(raw, dict) else None
        if m:
            out.append(m)
    if not out:
        raise ValueError("no Seedance video models in catalogue")
    return out


def get_models(refresh: bool = False, client: Optional[ModelHubClient] = None) -> List[Dict[str, Any]]:
    """Cached catalogue. Falls back to :data:`FALLBACK_MODELS` on any error or
    when ``MODELHUB_OFFLINE=1``."""
    global _cache, _cache_source
    with _lock:
        if _cache is not None and not refresh:
            return _cache
        if os.environ.get("MODELHUB_OFFLINE", "").strip() in ("1", "true", "yes"):
            _cache, _cache_source = list(FALLBACK_MODELS), "fallback (MODELHUB_OFFLINE)"
            return _cache
        try:
            base = os.environ.get("MODELHUB_BASE_URL", DEFAULT_BASE_URL)
            _cache, _cache_source = fetch_models(base_url=base, client=client), "live"
        except Exception as exc:  # network, JSON, shape... never break node loading
            print(f"[ModelHub] Could not fetch /v1/models ({exc}); using built-in model list.")
            _cache, _cache_source = list(FALLBACK_MODELS), "fallback"
        return _cache


def cache_source() -> str:
    return _cache_source


def reset_cache() -> None:
    global _cache, _cache_source
    with _lock:
        _cache, _cache_source = None, "none"


def find_model(model_id: str, models: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    for m in models if models is not None else get_models():
        if m["id"] == model_id:
            return m
    return None


def union(key: str, models: List[Dict[str, Any]], order: Optional[List[str]] = None) -> List[str]:
    seen: List[str] = []
    lowered = set()
    for m in models:
        for v in m.get(key) or []:
            if v.lower() not in lowered:
                lowered.add(v.lower())
                seen.append(v)
    if order:
        rank = {v.lower(): i for i, v in enumerate(order)}
        seen.sort(key=lambda v: rank.get(v.lower(), len(rank)))
    return seen


RESOLUTION_ORDER = ["480p", "720p", "1080p", "2k", "4k"]
RATIO_ORDER = ["9:16", "16:9", "1:1", "3:4", "4:3", "21:9", "auto", "adaptive"]


def validate_options(model: Dict[str, Any], ratio: str, resolution: str, duration: int) -> Dict[str, str]:
    """Check user choices against the model's accepted values. Returns the
    values to send (resolution case matched to the model, e.g. ``4k`` -> ``4K``).
    Empty accepted lists are treated as "unknown, allow anything"."""
    errors = []
    out = {"ratio": ratio, "resolution": resolution}
    if model.get("ratio") and ratio not in model["ratio"]:
        errors.append(f"ratio {ratio!r} not supported by {model['id']} (allowed: {', '.join(model['ratio'])})")
    if model.get("resolution"):
        match = [r for r in model["resolution"] if r.lower() == resolution.lower()]
        if match:
            out["resolution"] = match[0]
        else:
            errors.append(f"resolution {resolution!r} not supported by {model['id']} "
                          f"(allowed: {', '.join(model['resolution'])})")
    if model.get("duration") and str(int(duration)) not in model["duration"]:
        allowed = sorted(int(d) for d in model["duration"] if d.isdigit())
        rng = f"{allowed[0]}-{allowed[-1]}" if allowed else ", ".join(model["duration"])
        errors.append(f"duration {duration}s not supported by {model['id']} (allowed: {rng}s)")
    if errors:
        raise ValueError("; ".join(errors))
    return out
