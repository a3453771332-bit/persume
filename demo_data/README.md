# Frozen public-demo data

`frozen_home_credit_replay.csv` is a 1,000-case subset of the existing Home Credit B5.1 applicant-level audit artifact. It contains only precomputed, historical policy replay outputs. It does not contain training data, raw source tables, or a model that can be retrained.

The public runtime reads this file without importing research scripts. Full Evidence is a model reference, not ground truth. No TARGET values are included in the public runtime dataset.
