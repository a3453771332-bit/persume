"""One-off artifact exporter. Reads frozen audit output; never trains or tunes."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "b51_instance_adaptive_results" / "b51_applicant_level_audit_trail.csv"
DESTINATION = ROOT / "demo_data" / "frozen_home_credit_replay.csv"


def main() -> None:
    DESTINATION.parent.mkdir(exist_ok=True)
    columns = [
        "anonymous_applicant_id", "full_probability", "full_decision",
        "fixed_probability", "fixed_decision", "oracle_probability", "oracle_decision",
        "fixed_disagreement", "oracle_disagreement", "fixed_early_stop",
        "oracle_early_stop", "fixed_optional_units", "oracle_optional_units",
        "fixed_path", "oracle_path",
    ]
    frame = pd.read_csv(SOURCE, usecols=["decision_model", "target_delta", *columns], nrows=25_000)
    demo = frame.loc[(frame["decision_model"] == "Logistic Regression") & (frame["target_delta"] == 0.01), columns].head(1_000)
    if len(demo) != 1_000:
        raise RuntimeError(f"Expected 1,000 frozen replay rows, found {len(demo)}")
    demo.to_csv(DESTINATION, index=False)


if __name__ == "__main__":
    main()
