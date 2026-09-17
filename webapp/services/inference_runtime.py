"""Derived from frozen research outputs for public deployment only.

This module performs no training, threshold tuning, split generation, or
TARGET-based decisioning. It reads precomputed applicant-level replay values.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "demo_data" / "frozen_home_credit_replay.csv"
CONFIG = ROOT / "demo_data" / "policy_config.json"


@dataclass(frozen=True)
class PublicDemoRuntime:
    cases: pd.DataFrame
    threshold: float
    margin: float
    ranking: list[str]

    def case_id(self, position: int) -> str:
        return str(self.cases.iloc[position]["anonymous_applicant_id"])


def load_public_runtime() -> PublicDemoRuntime:
    if not DATA.exists() or not CONFIG.exists():
        raise FileNotFoundError("Public demo artifacts are missing. Commit demo_data/ to the deployment repository.")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    return PublicDemoRuntime(pd.read_csv(DATA), float(config["decision_threshold"]), float(config["default_margin"]), list(config["ranking"]))
