"""
Loads credit-scheme constants from credit_config.yaml (same directory).
Feel free to edit the YAML to tune the benchmark.
"""
from __future__ import annotations
import os, yaml, pathlib

_cfg_path = pathlib.Path(__file__).with_name("credit_config.yaml")
_cfg = yaml.safe_load(_cfg_path.read_text()) if _cfg_path.exists() else {}

BASE_CREDIT_DEFAULT = _cfg.get("base_credit_default", 10)
BASE_CREDIT_CORL = _cfg.get("base_credit_corl", 20)

WEEKLY_INC_DEFAULT = _cfg.get("weekly_increment_default", 5)
WEEKLY_INC_CORL = _cfg.get("weekly_increment_corl", 10)

UCB_TIMES_THRESHOLD = _cfg.get("ucb_times_threshold", 100)

# Owner whose policies have infinite credit (baseline servers)
INFINITE_CREDIT_OWNER = "pranavatreya@berkeley.edu"
