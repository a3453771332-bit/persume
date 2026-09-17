# Credit Evidence Policy Lab: 3-Minute Portfolio Brief

## The question

How much optional evidence can a credit workflow avoid requesting while preserving the decision made with all available evidence?

## What the study found

Across frozen experiments, evidence reduction spans **22%–61%**. The result is strongly dataset- and model-dependent. A simple fixed evidence order with a classification-margin stopping rule is a credible baseline; more elaborate adaptive ordering did not show stable incremental value over it.

## Why the evaluation is credible

- Decisions are made from pre-application evidence only. `TARGET` is revealed **after** the decision path is complete, solely for offline evaluation; it never drives evidence acquisition or the displayed decision.
- Validation covers IID, out-of-time, and structural stress settings. Bootstrap confidence intervals quantify uncertainty, while the Oracle policy is presented only as a diagnostic upper bound because it uses Full-Evidence look-ahead.
- The public deployment replays 1,000 frozen Home Credit test cases. It does not retrain a model, tune a policy, or accept a new applicant for scoring.

## What a recruiter can inspect

- **[Public demo](https://appapppy-zyezgezu6jgimid7p8wks6.streamlit.app/):** inspect a frozen case, sweep precomputed stopping margins, see the evidence-versus-agreement trade-off, and review disagreement cases.
- **[Local research application](webapp/app.py):** inspect the full research-connected workflow, including historical counterfactual replay, operational simulation, audit trail, and offline evaluation.

## Product reasoning, not just a model metric

The Operational Simulation labels its preset charts as **scenario trade-offs**. It does not claim that a preset is Pareto-dominant, because the charts compare illustrative operational assumptions rather than establish a universal best policy.

The Failure Explorer translates PD differences into illustrative expected-loss and net-contribution scenarios using `EAD × LGD` (exposure at default × loss given default). This makes the consequence of a disagreement visible in business terms without presenting the calculation as an actual realised loss or a production pricing model.

## Why the project has two release layers

The public and local applications are deliberately separated. The public layer ships anonymous, frozen replay outputs and minimal dependencies, which supports reproducibility without publishing raw source records, feature-engineering rules, or cached state-specific models. The local research layer retains those protected artifacts for controlled study and audit. This is a data-minimisation and reproducibility decision, not a shortcut in deployment.

## Boundary

This is research validation and decision-support demonstration work, not a live underwriting system. It is not designed for real credit decisions or regulatory adverse-action use.
