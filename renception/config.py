#!/usr/bin/env python3

"""Config loading. All pipeline parameters live in ``configs/*.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "fusion.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the pipeline config YAML (defaults to ``configs/fusion.yaml``)."""
    path = Path(path) if path is not None else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
