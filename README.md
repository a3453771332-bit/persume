# Progressive Decision Explorer

**Explore how decisions evolve as additional evidence becomes available.**

A public, interactive prototype for examining whether a decision workflow can request less optional information while preserving the recommendation reached with full information.

> **This prototype replays frozen out-of-sample decision trajectories and does not perform live credit scoring.**

[Open the public demo](https://appapppy-zyezgezu6jgimid7p8wks6.streamlit.app/) · [Methodology](#method-and-boundaries)

## What you can explore

The public demo is organised around three questions:

1. **Explore a decision** — choose one of 24 curated, anonymous held-out cases and request the next predefined evidence group step by step. See the current risk probability, provisional direction, information status, decision confidence, and final comparison with the full-information reference.
2. **Compare policies** — inspect the frozen trade-off between collecting less optional information and matching the full-information recommendation. The public interface translates technical settings into Lean, Balanced, and Conservative policy choices.
3. **Understand risk** — inspect where an early stop changed the full-information reference and translate PD differences into an illustrative expected-loss scenario.

The interactive sample includes early-stop, borderline, reject, decision-changed, agreement, and disagreement cases so that the workflow is visible rather than presented only as an aggregate result.

## Key result

In the Home Credit public demo, the **Balanced** frozen setting matches the full-information recommendation in approximately **99.7%** of the development replay while using about **2.9 of 5** optional evidence groups on average — roughly **43% fewer optional groups**.

Across the broader frozen experiments, evidence reduction ranged from **22% to 61%**. The attainable reduction depends materially on the dataset and model; it is not a universal property of the policy.

## What the project found

The project tested more complex adaptive ordering against a simpler frozen fixed-order policy. The complex approach did **not** deliver a stable advantage across the evaluated settings.

That negative/mixed result is retained deliberately:

```
Complex approaches
        ↓
Experimental comparison
        ↓
No stable advantage
        ↓
Simplified stopping rule
        ↓
Interactive prototype
```

The public prototype therefore demonstrates an explainable sequential evidence workflow rather than claiming a superior underwriting algorithm.

## Method and boundaries

- The public app reads only versioned, frozen deployment artifacts in `demo_data/`.
- The interactive Explorer uses **24 curated anonymous trajectories** from a frozen held-out Home Credit test split.
- Policy and impact views use the existing **1,000-case frozen replay** artifacts.
- The runtime does **not** import research scripts, train a model, tune a threshold, regenerate a split, call an evidence provider, or use `TARGET` in a decision workflow.
- Full information is a **model reference**, not ground truth.
- “Risk probability” and the illustrative impact view are experimental outputs, not a real credit decision or lending-loss estimate.
- The public app exposes scaled, anonymised display values rather than raw application records.

For research detail such as threshold selection, bootstrap analysis, cross-fitting, fidelity definitions, and the Oracle diagnostic, use the frozen research materials rather than inferring these details from the public UI.

## Run the public demo locally

```bash
pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

`streamlit_app.py` is the public entry point. It imports `webapp.public_app` and can start from the repository's public artifacts alone.

## Repository structure

```text
streamlit_app.py                 # Public Streamlit entry point
webapp/public_app.py             # Public Explorer interface
webapp/services/public_simulator.py
                                 # Read-only frozen trajectory runtime
demo_data/                       # Versioned public replay artifacts
webapp/app.py                    # Local research-connected application
RESEARCH_CODE_FREEZE.md          # Boundary protecting research artifacts
```

`webapp/app.py` is intentionally separate from the public demo. It requires the private/local research environment and is not required for public deployment.

## Research code freeze

Existing research code, splits, fitted-model logic, policy parameters, and published result artifacts are treated as read-only. Product presentation code lives in `webapp/`; the public runtime consumes frozen artifacts rather than rerunning the research pipeline.

See [RESEARCH_CODE_FREEZE.md](RESEARCH_CODE_FREEZE.md) for the boundary.

## Disclaimer

This is an experimental decision-support and portfolio prototype built from public historical data. It is **not** a production underwriting, loan-origination, adverse-action, or real-time credit-decision system.

