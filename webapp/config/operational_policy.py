"""Product-only operational scenarios. They never alter frozen research policy selection."""
PRESETS = {
    "Efficiency First": {"margin": 0.02, "max_units": 1, "manual_review_penalty": 2.0, "description": "Fewer requests; accepts more decision mismatch."},
    "Balanced": {"margin": 0.04, "max_units": 3, "manual_review_penalty": 2.0, "description": "Moderate evidence collection and decision fidelity."},
    "Fidelity First": {"margin": 0.08, "max_units": 5, "manual_review_penalty": 2.0, "description": "More evidence collection to approach the Full-Evidence reference."},
}
