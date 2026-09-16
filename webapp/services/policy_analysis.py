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
        'Approve': 'Risk Leakage：局部批准 → 全证据拒绝',
        'Reject': 'Lost Opportunity：局部拒绝 → 全证据批准'})
    result['局部预期损失 USD'] = result.fixed_probability * exposure * lgd
    result['全证据预期损失 USD'] = result.full_probability * exposure * lgd
    result['损失估计修正 USD'] = result['全证据预期损失 USD'] - result['局部预期损失 USD']
    # One-period, performing-loan net yield; same horizon as PD is assumed.
    result['全证据预期净贡献 USD'] = exposure * ((1-result.full_probability)*net_yield-result.full_probability*lgd)
    result['改按全证据决策的净贡献变化 USD'] = result['全证据预期净贡献 USD'] * result.fixed_decision.map({'Reject': 1, 'Approve': -1})
    return result
