"""Provider boundary for product simulation; no external service is called."""
from __future__ import annotations

from src.simulator import DemoEngine
from webapp.services.research_adapter import evidence_details


class HistoricalDatasetProvider:
    """Reveal only a requested, already-acquired historical evidence group."""

    def request_evidence(self, engine: DemoEngine, position: int, mask: int, unit: str) -> dict:
        bit = 1 << __import__("vc_credit_eswa_core").OPTIONAL_UNITS.index(unit)
        if not mask & bit:
            return {"available": False, "source": evidence_details(unit), "values": {}}
        state = engine.cache[mask]
        values = state["features"]["test"].iloc[position].to_dict()
        return {"available": True, "source": evidence_details(unit), "values": values}
