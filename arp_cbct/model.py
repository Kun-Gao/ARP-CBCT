"""Public construction interface for the released ARP-CBCT model."""

from __future__ import annotations

from typing import Any

from .config import get_default_config
from .models import ARPCBCTV4Balanced


ARPCBCT = ARPCBCTV4Balanced
ARP_CBCT = ARPCBCT


def build_model(config: dict[str, Any] | None = None) -> ARPCBCTV4Balanced:
    """Build the complete architecture used in the reported experiments."""

    model_config = get_default_config() if config is None else config
    architecture = model_config.get("architecture", "arp_cbct_v4_balanced")
    if architecture != "arp_cbct_v4_balanced":
        raise ValueError(
            "This model-only release contains arp_cbct_v4_balanced; "
            f"received architecture={architecture!r}."
        )
    return ARPCBCTV4Balanced(model_config)
