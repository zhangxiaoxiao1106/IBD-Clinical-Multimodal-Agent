"""读取 config/ 下的 YAML 配置。代码里读不到配置时用内置默认值。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def load_yaml(filename: str) -> Dict[str, Any]:
    path = _CONFIG_DIR / filename
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def agent_config() -> Dict[str, Any]:
    return load_yaml("agent_config.yaml")


def rag_config() -> Dict[str, Any]:
    return load_yaml("rag_config.yaml")
