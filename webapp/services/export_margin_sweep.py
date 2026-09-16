"""Read-only export of frozen development results; no training or tuning."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

if __name__ == '__main__':
    source = pd.read_csv(ROOT / 'b51_instance_adaptive_results/b51_fixed_policy_tune_selection.csv')
    source = source.loc[source.decision_model.eq('Logistic Regression')].copy()
    fields = ['threshold', 'development_acquired_optional_evidence_units', 'development_final_decision_disagreement', 'conditional_false_stable_stop_rate']
    frame = source[fields].drop_duplicates()
    assert not frame.threshold.duplicated().any()
    frame = frame.rename(columns={'threshold': 'margin', 'development_acquired_optional_evidence_units': 'evidence', 'conditional_false_stable_stop_rate': 'early_stop_failure'})
    frame['agreement'] = 1-frame.pop('development_final_decision_disagreement')
    frame['partition'] = 'development policy-tune (frozen source label)'
    frame.sort_values('margin').to_csv(ROOT / 'demo_data/development_margin_sweep.csv', index=False)
