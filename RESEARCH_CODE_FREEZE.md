# Research Code Freeze

The following artifacts are frozen research materials and are treated as read-only during public-product development:

- `vc_credit_eswa_core.py` and `vc_credit_progressive_privacy.py`
- All `run_b*.py`, `run_f*.py`, validation, audit, and report scripts
- Frozen split, fitting, calibration, policy-tuning, and threshold-selection logic
- Existing experiment result directories, CSV/JSON outputs, logs, figures, and checkpoints

The `webapp/` package is the product layer. The public app reads frozen deployment artifacts from `demo_data/`; it does not retrain models, tune policy settings, regenerate partitions, or write back to research outputs.

Manual product labels, layout, illustrative impact arithmetic, and historical replay controls belong to the public/product layer only.
