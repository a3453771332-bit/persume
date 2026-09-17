"""Read-only adapter mapping frozen research outputs to product concepts.

This module never writes to, tunes, or modifies research data, splits, models,
policies, or experiment artifacts. It delegates inference to ``src.simulator``,
which invokes the existing research functions unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.simulator import DemoEngine, build_engine
from webapp.config.evidence_mapping import EVIDENCE_SOURCES
import vc_credit_eswa_core as core


@dataclass
class ProductCase:
    position: int
    mode: str
    note: str


def load_engine() -> DemoEngine:
    return build_engine()


def product_decision(state: dict[str, Any], threshold: float, margin: float) -> str:
    """Product-only review band derived directly from the existing margin rule."""
    if not state["sufficient"]:
        return "Manual review"
    if state["probability"] < threshold - margin:
        return "Approve"
    return "Reject"


def reason_for_state(state: dict[str, Any]) -> str:
    if state["sufficient"]:
        return "Decision-boundary distance meets the configured stopping requirement"
    return "Current prediction is too close to the decision boundary"


def evidence_details(unit: str) -> dict[str, str]:
    return EVIDENCE_SOURCES[unit]


def nearest_historical_case(engine: DemoEngine, values: dict[str, Any]) -> int:
    """Choose a held-out prototype matching manual basics; no new score is trained."""
    frame = engine.raw_test
    age = -frame["DAYS_BIRTH"].astype(float) / 365.25
    years = (-frame["DAYS_EMPLOYED"].astype(float) / 365.25).clip(0, 50)
    income = frame["AMT_INCOME_TOTAL"].astype(float) / 12
    credit = frame["AMT_CREDIT"].astype(float)
    numeric = ((age - values["age"]) / 15) ** 2 + ((years - values["employment_years"]) / 10) ** 2 + ((income - values["monthly_income"]) / max(values["monthly_income"], 1)) ** 2 + ((credit - values["loan_amount"]) / max(values["loan_amount"], 1)) ** 2
    return int(np.argmin(numeric.to_numpy()))


def policy_lab(engine: DemoEngine, sample_size: int, margin: float) -> tuple[pd.DataFrame, dict[str, list[list[str]]]]:
    """Replay frozen test cases; TARGET is read only after decisions for offline metrics."""
    count = min(sample_size, engine.applicant_count)
    fixed = core.simulate_fixed_ranking(engine.cache, "test", engine.decision_threshold, engine.ranking, "margin", margin)
    # Existing B5.1 diagnostic. Its Full-Evidence look-ahead makes it explicitly
    # experimental/non-deployable; it is retained to expose genuine path diversity.
    oracle = core.simulate_oracle_greedy_adaptive(engine.cache, "test", engine.decision_threshold, engine.ranking, margin)
    all_units_cost = sum(EVIDENCE_SOURCES[unit]["illustrative_cost"] for unit in engine.ranking)
    target = engine.raw_test[core.TARGET].to_numpy(dtype=int)[:count]

    def summarize(name: str, probability: np.ndarray, masks: np.ndarray, paths: list[list[str]], disagreement: np.ndarray, source: str) -> dict[str, object]:
        probability, masks, disagreement, paths = probability[:count], masks[:count], disagreement[:count], paths[:count]
        approve = probability < engine.decision_threshold
        costs = np.array([sum(EVIDENCE_SOURCES[unit]["illustrative_cost"] for unit in path) for path in paths], dtype=float)
        if name == "Full Evidence": costs[:] = all_units_cost
        optional = np.array([int(value).bit_count() for value in masks], dtype=float)
        manual = np.array([not (abs(p - engine.decision_threshold) >= margin or mask == core.FULL_MASK) for p, mask in zip(probability, masks)])
        return {"Policy": name, "Average optional evidence units": optional.mean(), "Evidence reduction vs full": 1 - optional.mean() / len(engine.ranking), "Decision agreement with full": 1 - disagreement.mean(), "Disagreement rate": disagreement.mean(), "Approval rate": approve.mean(), "Portfolio bad rate (offline)": float(target[approve].mean()) if approve.any() else np.nan, "Manual review rate": manual.mean(), "Average acquisition cost*": costs.mean(), "Policy source": source}

    full_probability = engine.cache[core.FULL_MASK]["probabilities"]["test"]
    full_masks = np.full(engine.applicant_count, core.FULL_MASK, dtype=int)
    full_paths = [engine.ranking.copy() for _ in range(engine.applicant_count)]
    rows = [
        summarize("Full Evidence", full_probability, full_masks, full_paths, np.zeros(engine.applicant_count, dtype=bool), "Frozen full-evidence state"),
        summarize("Fixed Order + Margin", fixed["probability"], fixed["state_masks"], fixed["paths"], fixed["final_decision_disagreement_with_full"], "Frozen global order + existing Margin stopping"),
        summarize("Experimental Oracle Adaptive", oracle["probability"], oracle["state_masks"], oracle["paths"], oracle["final_decision_disagreement_with_full"], "Existing B5.1 Full-Evidence oracle diagnostic; not deployable"),
    ]
    return pd.DataFrame(rows), {"Fixed Order + Margin": fixed["paths"][:count], "Experimental Oracle Adaptive": oracle["paths"][:count]}


def path_diversity(paths: list[list[str]]) -> tuple[pd.DataFrame, list[str]]:
    rendered = ["Basic application → " + (" → ".join(evidence_details(unit)["name"] for unit in path) if path else "Stop") for path in paths]
    counts = pd.Series(rendered).value_counts()
    length = pd.Series([len(path) for path in paths]).value_counts().sort_index()
    first = pd.Series([path[0] if path else "Stop at initial assessment" for path in paths]).map(lambda item: evidence_details(item)["name"] if item in EVIDENCE_SOURCES else item).value_counts(normalize=True)
    table = pd.DataFrame({"Metric": ["Unique acquisition paths", "Most common path", "Average path length", "Stop after 0 evidence share"] + [f"First evidence: {name}" for name in first.index], "Value": [len(counts), counts.index[0], float(np.mean([len(path) for path in paths])), float((length.get(0, 0) / len(paths))) if paths else 0.0] + [float(value) for value in first.values]})
    return table, counts.index[:5].tolist()


def operational_portfolio(
    engine: DemoEngine, sample_size: int, decision_threshold: float, margin: float,
    max_units: int, manual_review_penalty: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Product scenario replay using frozen state predictions, without TARGET in policy flow."""
    count = min(sample_size, engine.applicant_count)
    rows: list[dict[str, object]] = []
    for position in range(count):
        mask, path = 0, []
        while True:
            probability = float(engine.cache[mask]["probabilities"]["test"][position])
            sufficient = abs(probability - decision_threshold) >= margin or mask == core.FULL_MASK
            if sufficient or len(path) >= max_units or mask == core.FULL_MASK:
                break
            mask, unit = engine.acquire_next(mask)
            path.append(unit)
        model_approve = probability < decision_threshold
        review = not sufficient
        product = "Manual review" if review else ("Approve" if model_approve else "Reject")
        full_probability = float(engine.cache[core.FULL_MASK]["probabilities"]["test"][position])
        full_approve = full_probability < decision_threshold
        cost = sum(EVIDENCE_SOURCES[unit]["illustrative_cost"] for unit in path)
        delay = sum(EVIDENCE_SOURCES[unit]["illustrative_delay_points"] for unit in path)
        rows.append({"Case ID": engine.applicant_id(position), "Current PD": probability, "Decision threshold": decision_threshold, "Boundary distance": abs(probability - decision_threshold), "Evidence acquired": len(path), "Evidence path": " → ".join(evidence_details(unit)["name"] for unit in path) if path else "Basic application only", "Model prediction": "Approve" if model_approve else "Reject", "Product decision": product, "Manual review": review, "Initial evidence sufficient": abs(float(engine.cache[0]["probabilities"]["test"][position]) - decision_threshold) >= margin, "Additional evidence required": bool(path), "Agreement with full": model_approve == full_approve, "Approve to reject disagreement": model_approve and not full_approve, "Reject to approve disagreement": not model_approve and full_approve, "Evidence cost points*": cost, "Delay points*": delay, "Applicant friction index*": len(path) + manual_review_penalty * int(review), "Reason for review": "Maximum evidence units reached while evidence remains insufficient" if review and len(path) >= max_units else ("Near decision boundary / evidence insufficient" if review else "")})
    detail = pd.DataFrame(rows)
    # Explicitly appended only after every applicant-level policy decision is fixed.
    detail["Historical TARGET"] = engine.raw_test[core.TARGET].to_numpy(dtype=int)[:count]
    model_approve = detail["Model prediction"].eq("Approve")
    automatic = ~detail["Manual review"]
    metrics = {"Total applications": len(detail), "Decision with initial evidence": int(detail["Initial evidence sufficient"].sum()), "Additional evidence required": int(detail["Additional evidence required"].sum()), "Manual review required": int(detail["Manual review"].sum()), "Automatically approved": int((automatic & model_approve).sum()), "Automatically rejected": int((automatic & ~model_approve).sum()), "Approval rate": float((automatic & model_approve).mean()), "Reject rate": float((automatic & ~model_approve).mean()), "Manual review rate": float(detail["Manual review"].mean()), "Decision agreement with full": float(detail["Agreement with full"].mean()), "Approve to reject disagreement": float(detail["Approve to reject disagreement"].mean()), "Reject to approve disagreement": float(detail["Reject to approve disagreement"].mean()), "Average optional evidence units": float(detail["Evidence acquired"].mean()), "Evidence reduction vs full": float(1 - detail["Evidence acquired"].mean() / len(engine.ranking)), "Average evidence cost points*": float(detail["Evidence cost points*"].mean()), "Average delay points*": float(detail["Delay points*"].mean()), "Applicant friction index*": float(detail["Applicant friction index*"].mean()), "Historical portfolio bad rate (offline)": float(detail["Historical TARGET"].mean()), "Approved portfolio bad rate (offline)": float(detail.loc[automatic & model_approve, "Historical TARGET"].mean()) if (automatic & model_approve).any() else np.nan}
    return metrics, detail


def counterfactual_evidence(engine: DemoEngine, position: int, mask: int, threshold: float, margin: float) -> pd.DataFrame:
    """Single-case historical counterfactuals. No state or audit object is mutated."""
    current = engine.state(position, mask, margin)
    rows = []
    for rank, unit in enumerate(current["remaining"], start=1):
        bit = 1 << core.OPTIONAL_UNITS.index(unit)
        after = engine.state(position, mask | bit, margin)
        rows.append({"Internal ID": unit, "Evidence source": evidence_details(unit)["name"], "Current priority rank": rank, "PD after": after["probability"], "Model decision after": "Approve" if after["probability"] < threshold else "Reject", "Boundary improvement": after["margin"] - current["margin"], "Would stop?": after["sufficient"], "Decision changed?": (after["probability"] < threshold) != (current["probability"] < threshold), "Agreement before": (current["probability"] < threshold) == (current["full_probability"] < threshold), "Agreement after": (after["probability"] < threshold) == (after["full_probability"] < threshold), "Agreement change": int((after["probability"] < threshold) == (after["full_probability"] < threshold)) - int((current["probability"] < threshold) == (current["full_probability"] < threshold)), "Cost points*": evidence_details(unit)["illustrative_cost"], "Delay points*": evidence_details(unit)["illustrative_delay_points"], "Privacy level": evidence_details(unit)["privacy"], "Recommended?": rank == 1})
    return pd.DataFrame(rows)


def failure_explorer(engine: DemoEngine, sample_size: int, margin: float) -> pd.DataFrame:
    """Frozen Fixed-Margin replay; decisions are made before any optional offline fields."""
    count = min(sample_size, engine.applicant_count)
    fixed = core.simulate_fixed_ranking(engine.cache, "test", engine.decision_threshold, engine.ranking, "margin", margin)
    rows = []
    for i in range(count):
        progressive = fixed["probability"][i] < engine.decision_threshold
        full = fixed["full_probability"][i] < engine.decision_threshold
        disagreement = progressive != full
        if progressive and not full: failure = "Risk Leakage (Approve → Reject)"
        elif not progressive and full: failure = "Lost Opportunity (Reject → Approve)"
        else: failure = "Agreement"
        path = fixed["paths"][i]; missing = [unit for unit in engine.ranking if unit not in path]
        # Existing margin-only product review is not triggered after a normal
        # sufficient stop; this intentionally reveals its limited capture scope.
        review = not bool(fixed["early_stop"][i]) and len(path) < len(engine.ranking)
        rows.append({"Case ID": engine.applicant_id(i), "Selected policy": "Fixed Order + Margin", "Progressive decision": "Approve" if progressive else "Reject", "Full Evidence decision": "Approve" if full else "Reject", "Current PD": fixed["probability"][i], "Full Evidence PD": fixed["full_probability"][i], "Evidence units acquired": len(path), "Stopping step": len(path), "Failure type": failure, "Early stop failure": bool(fixed["early_stop"][i] and disagreement), "Review triggered?": review, "Potentially preventable by review": bool(review and disagreement), "Key missing evidence": evidence_details(missing[0])["name"] if missing else "None", "Path": path, "Final mask": int(fixed["state_masks"][i])})
    return pd.DataFrame(rows)


def failure_timeline(engine: DemoEngine, case_id: str, margin: float) -> pd.DataFrame:
    position = next(i for i in range(engine.applicant_count) if engine.applicant_id(i) == case_id)
    fixed = core.simulate_fixed_ranking(engine.cache, "test", engine.decision_threshold, engine.ranking, "margin", margin)
    path = fixed["paths"][position]; mask = 0; rows=[]
    for step, unit in enumerate([None, *path]):
        if unit is not None: mask |= 1 << core.OPTIONAL_UNITS.index(unit)
        state = engine.state(position, mask, margin)
        rows.append({"Step": step, "Evidence added": "Basic application" if unit is None else evidence_details(unit)["name"], "PD": state["probability"], "Model decision": "Approve" if state["probability"] < engine.decision_threshold else "Reject", "Boundary distance": state["margin"], "Evidence sufficiency": "Sufficient" if state["sufficient"] else "Insufficient"})
    full = engine.state(position, core.FULL_MASK, margin)
    rows.append({"Step": "Full reference", "Evidence added": "Full Evidence reference", "PD": full["probability"], "Model decision": "Approve" if full["probability"] < engine.decision_threshold else "Reject", "Boundary distance": full["margin"], "Evidence sufficiency": "Reference only"})
    return pd.DataFrame(rows)


def failure_policy_comparison(engine: DemoEngine, case_id: str, margin: float) -> pd.DataFrame:
    """Same applicant, frozen fixed policy versus existing experimental oracle diagnostic."""
    pos = next(i for i in range(engine.applicant_count) if engine.applicant_id(i) == case_id)
    fixed = core.simulate_fixed_ranking(engine.cache, "test", engine.decision_threshold, engine.ranking, "margin", margin)
    oracle = core.simulate_oracle_greedy_adaptive(engine.cache, "test", engine.decision_threshold, engine.ranking, margin)
    full_p = float(engine.cache[core.FULL_MASK]["probabilities"]["test"][pos]); full_decision = full_p < engine.decision_threshold
    rows=[]
    for name, simulation, note in [("Fixed Margin",fixed,"Deployable frozen global order"),("Experimental Oracle Adaptive",oracle,"B5.1 diagnostic; Full-Evidence look-ahead, not deployable")]:
        p=float(simulation["probability"][pos]); decision=p < engine.decision_threshold
        rows.append({"Policy":name,"Stop step":len(simulation["paths"][pos]),"Evidence count":len(simulation["paths"][pos]),"Final decision":"Approve" if decision else "Reject","Agreement with Full Evidence":decision==full_decision,"Manual review":False,"Outcome":"Matched" if decision==full_decision else "Missed","Policy note":note})
    rows.insert(0,{"Policy":"Full Evidence reference","Stop step":len(engine.ranking),"Evidence count":len(engine.ranking),"Final decision":"Approve" if full_decision else "Reject","Agreement with Full Evidence":True,"Manual review":False,"Outcome":"Reference","Policy note":"Model reference, not ground truth"})
    return pd.DataFrame(rows)
