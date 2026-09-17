# Credit Evidence Policy Lab

**Problem:** Can a credit decision request less optional evidence without changing the decision that would have been made after reviewing everything?

**Core finding:** Evidence reduction ranges from **22% to 61%** across the studied settings, and the attainable reduction is **highly dependent on the dataset and model** rather than being a universal property of an evidence-acquisition policy.

![Existing evidence-fidelity frontier from the frozen research results](成果展示_F2/evidence_fidelity_frontier.png)

- **Methodological rigour.** The study evaluates IID, out-of-time, and structural stress settings; reports bootstrap confidence intervals; and uses an Oracle upper bound only as a diagnostic reference.
- **An honest mixed result.** More complex adaptive ordering did not consistently outperform a simple frozen fixed-order policy, so the project reports the negative result rather than treating adaptivity as an assumption.
- **Product judgement.** This is a research validation tool, not an underwriting product: it replays historical evidence paths to examine decision stability and does not make live credit decisions.

## Two ways to inspect the project

- **[Public demo](https://appapppy-zyezgezu6jgimid7p8wks6.streamlit.app/)** — a cloud-safe, read-only replay of 1,000 frozen Home Credit test cases. It shows policy trade-offs, disagreement cases, and illustrative economic scenarios. It does not accept new applicants, expose raw data, train models, tune thresholds, or use `TARGET` in a decision.
- **[Local research application](webapp/app.py)** — the research-connected Streamlit interface for the original local environment. It includes applicant exploration, historical counterfactual replay, operational simulation, audit trails, and offline evaluation. It requires the private raw datasets and frozen research artifacts, so it is intentionally not part of the public deployment.

For a three-minute portfolio summary, see [PORTFOLIO.md](PORTFOLIO.md). The frozen-artifact boundary is defined in [RESEARCH_CODE_FREEZE.md](RESEARCH_CODE_FREEZE.md).

## Run the public demo locally

```bash
pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

`app.py` and `streamlit_app.py` are public-demo entry points. The deployed runtime reads only versioned files in `demo_data/`; it does not import the research pipeline. The local research application is `webapp/app.py` and has separate private-data requirements.

## Interpretation boundary

“Margin” means the classification distance `abs(PD − decision threshold)`, not a deposit, collateral requirement, or financial charge. An evidence unit is one predefined optional group of historical information. “Oracle Adaptive” is a non-deployable diagnostic that can look ahead to the Full-Evidence reference; it is an upper-bound comparison, not a recommendation.

This project is an educational decision-support demonstration, not a production loan-origination, underwriting, or adverse-action system. It must not be used to make real credit decisions.
