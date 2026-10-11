"""Cloud-safe state helper for frozen progressive replay only."""
from __future__ import annotations
import base64
import gzip
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "demo_data"

@dataclass(frozen=True)
class ProgressiveReplay:
    trajectories: pd.DataFrame
    templates: pd.DataFrame
    threshold: float
    margin: float
    ranking: list[str]
    @property
    def case_ids(self) -> list[str]: return self.templates["anonymous_applicant_id"].tolist()
    def profile(self, case_id: str) -> pd.Series: return self.templates.loc[self.templates["anonymous_applicant_id"].eq(case_id)].iloc[0]
    def states(self, case_id: str) -> pd.DataFrame: return self.trajectories.loc[self.trajectories["anonymous_applicant_id"].eq(case_id)].sort_values("step").reset_index(drop=True)

def _read_frame(name: str) -> pd.DataFrame:
    csv_path = DEMO / name
    packed_path = csv_path.with_suffix(csv_path.suffix + ".gz.b64")
    if csv_path.exists(): return pd.read_csv(csv_path)
    payload = gzip.decompress(base64.b64decode(packed_path.read_text(encoding="ascii")))
    return pd.read_csv(BytesIO(payload))

def load_progressive_replay() -> ProgressiveReplay:
    config_path = DEMO / "progressive_replay_config.json"
    if not config_path.exists(): raise FileNotFoundError("Progressive replay artifacts are missing.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return ProgressiveReplay(_read_frame("frozen_progressive_replay.csv").fillna(""), _read_frame("applicant_templates.csv").fillna("Not recorded"), float(config["decision_threshold"]), float(config["default_margin"]), list(config["ranking"]))
