"""Cloud-safe public demo; uses only demo_data frozen artifacts."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from webapp.services.inference_runtime import load_public_runtime
from webapp.services.policy_analysis import load_sweep, pareto_points, economic_cases


@st.cache_resource
def runtime():
    return load_public_runtime()


def run() -> None:
    st.set_page_config(page_title="Credit Evidence Policy Lab", layout="wide")
    rt = runtime()
    st.title("Credit Evidence Policy Lab")
    st.markdown("### An AI-assisted decision prototype for reducing unnecessary information requests while keeping credit-approval outcomes as consistent as possible.")
    st.markdown("`AI-assisted development` &nbsp; `Decision experiments` &nbsp; `Interactive prototype`")
    st.write("This public demo turns a research question into an inspectable workflow: review a historical decision, see what information was used, and examine the trade-off between asking for less and keeping the same outcome.")

    with st.container(border=True):
        st.subheader("How the project was built")
        st.write("Started with a complex dynamic gating approach → experiments found it was not consistently better than a simpler policy → simplified the design deliberately → integrated the result into an explainable prototype.")
        st.caption("The point is not to maximise technical complexity. It is to test an idea honestly, keep the parts that hold up, and make the resulting decision process understandable.")
        st.divider()
        st.subheader("My role and how AI contributed")
        st.write("I was responsible for problem definition, data analysis, experiment design, result interpretation, solution simplification, and prototype integration. I used ChatGPT, Codex, and Gemini to assist with code generation, debugging, and interface iteration.")

    with st.expander("Technical details and research boundary"):
        st.write("This deployment is a read-only replay of 1,000 frozen Home Credit test cases. It does not train models, tune a policy, score new applicants, or use observed outcomes in a decision. Advanced research terms and diagnostics are kept in the later tabs so the first screen stays focused on the decision workflow.")

    explore, compare, failures, assumptions = st.tabs(["Review a decision", "Compare approaches", "Review disagreement cases", "Assumptions"])
    with explore:
        st.subheader("What the system knew, why it stopped, and what happened next")
        pos = st.selectbox("Historical replay case", range(len(rt.cases)), format_func=rt.case_id)
        row = rt.cases.iloc[pos]
        a, b, c = st.columns(3)
        a.metric("Current estimated default risk", f"{row.fixed_probability:.1%}")
        b.metric("Final approval recommendation", row.fixed_decision)
        c.metric("Agreement with the full-information result", "Yes" if not row.fixed_disagreement else "No")
        st.write(f"**Information acquired:** {row.fixed_path or 'The initial application information was sufficient.'}")
        st.write("**Why the system stopped:** the frozen policy stopped once the information already collected was sufficient for a stable approval recommendation under its precomputed rule.")
        st.write(f"**Full-information reference:** estimated default risk {row.full_probability:.1%}. This is a model reference for comparison, not ground truth.")
        st.caption("The system requests another information group only when the current evidence is not yet sufficient for a stable decision. This public artifact preserves final frozen replay states and does not expose source records or permit new inference / counterfactual recomputation.")
    with compare:
        st.subheader("Precomputed margin sweep: development-set evidence-agreement trade-off")
        sweep = load_sweep()
        st.caption("Margin is the classification distance from the decision boundary, not a deposit or a probability confidence interval.")
        margin = st.slider('Classification margin', min_value=float(sweep.margin.min()), max_value=float(sweep.margin.max()), value=0.05, step=0.01)
        point = sweep.loc[sweep.margin.eq(margin)].iloc[0]
        left, right = st.columns(2)
        left.metric('Development-set average optional evidence units', f'{point.evidence:.3f}')
        right.metric('Development-set agreement with Full Evidence', f'{point.agreement:.3%}')
        frontier = pareto_points(sweep)
        chart = px.scatter(sweep, x='evidence', y='agreement', hover_data=['margin', 'early_stop_failure'], labels={'evidence': 'Average optional evidence units', 'agreement': 'Agreement with Full-Evidence decision'}, template='plotly_white')
        chart.add_scatter(x=frontier.evidence, y=frontier.agreement, mode='lines+markers', name='Empirical Pareto frontier among scanned points')
        chart.add_scatter(x=[point.evidence], y=[point.agreement], mode='markers', marker={'size': 15, 'symbol': 'star'}, name='Selected margin')
        replay_reference = pd.DataFrame([
            {'Policy': 'Fixed Margin replay', 'evidence': rt.cases.fixed_optional_units.mean(), 'agreement': 1-rt.cases.fixed_disagreement.mean()},
            {'Policy': 'Oracle Adaptive replay', 'evidence': rt.cases.oracle_optional_units.mean(), 'agreement': 1-rt.cases.oracle_disagreement.mean()},
        ])
        chart.add_scatter(x=replay_reference.evidence, y=replay_reference.agreement, mode='markers+text', text=replay_reference.Policy, textposition='top center', marker={'size': 11, 'symbol': 'diamond-open'}, name='Frozen test-replay reference points')
        chart.update_yaxes(tickformat='.2%', range=[max(0, sweep.agreement.min()-.003), min(1.001, sweep.agreement.max()+.003)])
        st.plotly_chart(chart, width='stretch')
        increments = frontier[['margin', 'evidence', 'agreement']].copy()
        increments['Agreement gain per additional evidence unit (percentage points)'] = 100*increments.agreement.diff()/increments.evidence.diff()
        st.dataframe(increments, hide_index=True)
        st.caption('All 30 margin candidates come from a frozen development-set policy-tuning sweep; the slider selects a corresponding precomputed point. The line joins only observed non-dominated points and does not imply that intermediate thresholds were evaluated. Diamonds are separate frozen test-replay references, not estimates from the development curve. Each evidence unit is treated as equal-cost.')
        st.subheader('Separate view: 1,000 frozen test replays')
        frame = rt.cases
        results = pd.DataFrame([
            {"Policy": "Full Evidence reference", "Average optional evidence": 5, "Agreement with Full Evidence": 1.0},
            {"Policy": "Fixed Margin", "Average optional evidence": frame.fixed_optional_units.mean(), "Agreement with Full Evidence": 1 - frame.fixed_disagreement.mean()},
            {"Policy": "Experimental Oracle Adaptive", "Average optional evidence": frame.oracle_optional_units.mean(), "Agreement with Full Evidence": 1 - frame.oracle_disagreement.mean()},
        ])
        st.dataframe(results, width="stretch", hide_index=True)
        results['Disagreements per 1,000 cases'] = ((1-results['Agreement with Full Evidence'])*len(frame)).round().astype(int)
        chart = px.bar(results, x='Policy', y='Disagreements per 1,000 cases', text='Disagreements per 1,000 cases', template='plotly_white')
        st.plotly_chart(chart, width="stretch")
        st.caption("Oracle Adaptive is an existing diagnostic that uses Full-Evidence look-ahead; it is not deployable.")
    with failures:
        frame = rt.cases.copy()
        exposure = st.number_input('Illustrative average EAD (USD)', min_value=0.0, value=10000.0, step=1000.0)
        lgd = st.slider('Illustrative LGD (loss given default)', 0.0, 1.0, 0.45, 0.01)
        net_yield = st.slider('Illustrative net yield for a repaid loan', 0.0, 0.5, 0.10, 0.01)
        failed = economic_cases(frame, exposure, lgd, net_yield)
        st.metric("Frozen Fixed-Margin disagreements", len(failed))
        missed_risk = int(failed.fixed_decision.eq('Approve').sum())
        lost = int(failed.fixed_decision.eq('Reject').sum())
        st.write(f'Missed Risk: {missed_risk} / {len(frame)}; Lost Opportunity: {lost} / {len(frame)}. Directions are Fixed Margin decision → Full-Evidence decision.')
        if missed_risk == 0:
            st.info('In this dataset, all Fixed-Margin disagreements are Lost Opportunity cases; no risk-underestimation cases were observed. This finding applies only to these 1,000 frozen replays.')
        elif lost == 0:
            st.info('In this dataset, all Fixed-Margin disagreements are risk-underestimation cases; no Lost Opportunity cases were observed. This finding applies only to these 1,000 frozen replays.')
        choice = st.selectbox('Disagreement direction', ['All', 'Missed Risk', 'Lost Opportunity'])
        if choice != 'All':
            failed = failed.loc[failed.fixed_decision.eq('Approve' if choice == 'Missed Risk' else 'Reject')]
        failed = failed.sort_values('Expected Loss Impact Magnitude ($)', ascending=False)
        st.dataframe(failed[["anonymous_applicant_id", "Failure type", "fixed_probability", "full_probability", 'Estimated Expected Loss Impact ($)', 'Expected Loss Impact Magnitude ($)', 'Fixed expected loss (USD)', 'Full-Evidence expected loss (USD)', 'Full-Evidence expected net contribution (USD)', 'Net contribution change from using Full Evidence (USD)', "fixed_path"]], width="stretch", hide_index=True)
        st.caption('Illustrative scenario: Estimated Expected Loss Impact = (Fixed-Margin PD − Full-Evidence PD) × EAD × LGD. A positive value means the Fixed-Margin estimate is higher; a negative value means it is lower. The table is sorted by absolute dollar impact. Expected net contribution = EAD × [(1−PD) × net yield − PD × LGD]; rejected applications have zero net contribution. Amounts are simplified estimates, not a real risk model or realised losses.')
    with assumptions:
        st.markdown('''**Economic scenario parameters in Failure Explorer:** default EAD (average exposure at default) is **$10,000**, LGD (loss given default) is **45%**, and net yield for a repaid loan is **10%**. These inputs only change the dollar translation; they do not change PDs, policy paths, or any research result.''')
        st.markdown('''**How the frozen path was ordered:** the original study used a stratified 75/25 development split and Logistic Regression validation ROC-AUC to evaluate stepwise backward deletion of complete evidence units. At each round, it selected the removal candidate with the highest retained AUC, using PR-AUC and unit name to break ties, subject to a maximum 0.01 AUC loss relative to Full Evidence. The final order places retained units first and remaining units afterward, preserving original within-group order. This is neither applicant-specific greedy information-gain ordering nor a complete ranking by marginal AUC gain. Source: `vc_credit_eswa_core.py::select_fixed_minimal_units`; B5.1 invokes the same frozen method.

**Why public users cannot score manual inputs or run real-time counterfactuals:** the deployment intentionally publishes only anonymous final replays, frozen paths, and minimal dependencies. It excludes raw application features, feature-engineering rules, per-evidence-state model caches, and training artifacts. Keeping external inputs separate avoids producing PDs that cannot be reproduced from the frozen research environment, and maintains the data-minimisation boundary. These controls remain available only in the local research application. The dollar scenario is arithmetic on published PDs, not a new model counterfactual.

**What the frontier means:** it marks only the empirical non-dominated set among scanned candidates. A development-set frontier does not guarantee test-set optimality. Oracle uses Full-Evidence look-ahead and is diagnostic only, not a deployable candidate.''')
        st.markdown("""- This public demo reads a 1,000-case frozen Home Credit historical replay.
- It does not import research scripts, train a model, tune a threshold, or read TARGET.
- Full Evidence is a model reference, not ground truth.
- Counterfactual simulation and manual-input scoring remain available only in the local research-connected application because their raw evidence-state artifacts are not published.""")

