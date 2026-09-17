"""Product-only arithmetic over frozen replay and development sweep artifacts."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def load_sweep():
    return pd.read_csv(ROOT / 'demo_data' / 'development_margin_sweep.csv')


def pareto_points(frame):
    ordered = frame.sort_values(['evidence', 'agreement'], ascending=[True, False])
    ordered = ordered.drop_duplicates('evidence')
    return ordered.loc[ordered.agreement.gt(ordered.agreement.cummax().shift(fill_value=-1))]


def economic_cases(frame, exposure, lgd, net_yield):
    result = frame.loc[frame.fixed_disagreement].copy()
    result['Failure type'] = result.fixed_decision.map({
        'Approve': 'Missed Risk: Fixed Approve → Full-Evidence Reject',
        'Reject': 'Lost Opportunity: Fixed Reject → Full-Evidence Approve'})
    result['Fixed expected loss (USD)'] = result.fixed_probability * exposure * lgd
    result['Full-Evidence expected loss (USD)'] = result.full_probability * exposure * lgd
    result['Estimated Expected Loss Impact ($)'] = (
        (result.fixed_probability - result.full_probability) * exposure * lgd
    )
    result['Expected Loss Impact Magnitude ($)'] = result['Estimated Expected Loss Impact ($)'].abs()
    # One-period, performing-loan net yield; same horizon as PD is assumed.
    result['Full-Evidence expected net contribution (USD)'] = exposure * ((1-result.full_probability)*net_yield-result.full_probability*lgd)
    result['Net contribution change from using Full Evidence (USD)'] = result['Full-Evidence expected net contribution (USD)'] * result.fixed_decision.map({'Reject': 1, 'Approve': -1})
    return result
