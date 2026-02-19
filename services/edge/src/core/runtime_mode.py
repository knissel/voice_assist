"""Runtime mode helpers for edge compute routing."""
import logging
import os

logger = logging.getLogger(__name__)

EDGE_MODE_HYBRID = "hybrid"
EDGE_MODE_LOCAL = "local"
_VALID_EDGE_MODES = {EDGE_MODE_HYBRID, EDGE_MODE_LOCAL}


def resolve_edge_mode(raw_value: str | None = None) -> str:
    """
    Resolve EDGE_MODE from env or explicit value.

    Valid values:
    - hybrid: remote STT first, local fallback
    - local: local-only STT/TTS routing (no 5090 dependency)
    """
    raw = raw_value if raw_value is not None else os.getenv("EDGE_MODE", EDGE_MODE_HYBRID)
    mode = (raw or "").strip().lower()
    if mode in _VALID_EDGE_MODES:
        return mode
    logger.warning("Invalid EDGE_MODE=%r; defaulting to %s", raw, EDGE_MODE_HYBRID)
    return EDGE_MODE_HYBRID
