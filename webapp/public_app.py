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
    st.markdown("### An AI-assisted decision prototype that tests whether a credit workflow can ask for less information without changing the final recommendation.")
    st.markdown("`AI-assisted build` &nbsp; `Experiment-led iteration` &nbsp; `Interactive historical replay`")
    st.write("Collecting every document upfront creates delay, applicant friction, privacy exposure, and operational cost. This prototype explores when the system can stop early—and when it should ask for more information.")
    st.info("Experimental historical replay — not a live underwriting system.")
    st.caption("Across frozen experiments, information reduction ranged from 22% to 61%, depending on the dataset and model. Start by choosing a replay case below.")

    explore, compare, failures, assumptions = st.tabs(["Try the replay", "See the trade-offs", "Review impact scenarios", "Technical details"])
    with explore:
        st.subheader("1. Choose a historical case")
        pos = st.selectbox("Replay case", range(len(rt.cases)), format_func=rt.case_id)
        row = rt.cases.iloc[pos]
        optional_units = int(row.fixed_optional_units)
        stop_reason = (
            f"The replay used all {optional_units} optional information groups, so no further reduction was available for this case."
            if optional_units >= 5 else
            f"The replay stopped after {optional_units} optional information group{'s' if optional_units != 1 else ''} because the frozen policy judged the recommendation stable."
        )
        st.subheader("2. Review the decision summary")
        a, b, c = st.columns(3)
        a.metric("Current estimated default risk", f"{row.fixed_probability:.1%}")
        b.metric("Final approval recommendation", row.fixed_decision)
        c.metric("Agreement with the full-information result", "Yes" if not row.fixed_disagreement else "No")
        st.write(f"**Information acquired:** {row.fixed_path or 'The initial application information was sufficient.'}")
        st.write(f"**Why the system stopped:** {stop_reason}")
        st.write(f"**Full-information reference:** estimated default risk {row.full_probability:.1%}. This is a model reference for comparison, not ground truth.")
        st.caption("In the original workflow, another information group is requested only when the current evidence is not sufficient for a stable decision. This public demo replays the final frozen path rather than scoring a new applicant or recomputing intermediate steps.")
    with compare:
        st.subheader("What changes when we ask for less information?")
        st.write("Use the control below to explore the trade-off between requesting fewer optional information groups and matching the result reached with all available information.")
        sweep = load_sweep()
        margin = st.slider('How cautious should the stop rule be?', min_value=float(sweep.margin.min()), max_value=float(sweep.margin.max()), value=0.05, step=0.01)
        point = sweep.loc[sweep.margin.eq(margin)].iloc[0]
        left, right = st.columns(2)
        left.metric('Average optional information groups requested', f'{point.evidence:.3f}')
        right.metric('Agreement with the full-information result', f'{point.agreement:.3%}')
        frontier = pareto_points(sweep)
        chart = px.scatter(sweep, x='evidence', y='agreement', hover_data=['margin'], labels={'evidence': 'Average optional information groups requested', 'agreement': 'Agreement with full-information result'}, template='plotly_white')
        chart.add_scatter(x=frontier.evidence, y=frontier.agreement, mode='lines+markers', name='Best observed trade-offs')
        chart.add_scatter(x=[point.evidence], y=[point.agreement], mode='markers', marker={'size': 15, 'symbol': 'star'}, name='Your selected setting')
        replay_reference = pd.DataFrame([
            {'Policy': 'Simplified stopping-rule replay', 'evidence': rt.cases.fixed_optional_units.mean(), 'agreement': 1-rt.cases.fixed_disagreement.mean()},
            {'Policy': 'Diagnostic full-information look-ahead', 'evidence': rt.cases.oracle_optional_units.mean(), 'agreement': 1-rt.cases.oracle_disagreement.mean()},
        ])
        chart.add_scatter(x=replay_reference.evidence, y=replay_reference.agreement, mode='markers+text', text=replay_reference.Policy, textposition='top center', marker={'size': 11, 'symbol': 'diamond-open'}, name='Frozen test-replay reference points')
        chart.update_yaxes(tickformat='.2%', range=[max(0, sweep.agreement.min()-.003), min(1.001, sweep.agreement.max()+.003)])
        st.plotly_chart(chart, width='stretch')
        st.caption('The curve uses precomputed development-set settings. Diamonds are separate frozen test-replay references, not estimates from the curve. Each information group is treated as equal-cost for this illustration.')
        st.subheader('What happened in 1,000 frozen test replays')
        frame = rt.cases
        results = pd.DataFrame([
            {"Policy": "Full-information reference", "Average optional evidence": 5, "Agreement with Full Evidence": 1.0},
            {"Policy": "Simplified stopping rule", "Average optional evidence": frame.fixed_optional_units.mean(), "Agreement with Full Evidence": 1 - frame.fixed_disagreement.mean()},
            {"Policy": "Diagnostic look-ahead", "Average optional evidence": frame.oracle_optional_units.mean(), "Agreement with Full Evidence": 1 - frame.oracle_disagreement.mean()},
        ])
        st.dataframe(results, width="stretch", hide_index=True)
        results['Disagreements per 1,000 cases'] = ((1-results['Agreement with Full Evidence'])*len(frame)).round().astype(int)
        chart = px.bar(results, x='Policy', y='Disagreements per 1,000 cases', text='Disagreements per 1,000 cases', template='plotly_white')
        st.plotly_chart(chart, width="stretch")
        with st.expander("Technical details for this comparison"):
            st.write("The slider changes the classification margin: a technical stopping-rule sensitivity parameter, not a deposit or a probability confidence interval.")
            increments = frontier[['margin', 'evidence', 'agreement']].copy()
            increments['Agreement gain per additional information group (percentage points)'] = 100*increments.agreement.diff()/increments.evidence.diff()
            st.dataframe(increments, hide_index=True)
            st.caption("The diagnostic full-information look-ahead is not deployable. The line connects only observed non-dominated settings and does not imply that intermediate settings were evaluated.")
    with failures:
        st.subheader("What did the disagreements mean in an illustrative business scenario?")
        st.write("These cases compare the simplified replay with the full-information reference. Dollar amounts are illustrative estimates, not realised profit or loss.")
        frame = rt.cases.copy()
        with st.expander("Adjust the illustrative financial assumptions"):
            exposure = st.number_input('Average exposure at default (USD)', min_value=0.0, value=10000.0, step=1000.0)
            lgd = st.slider('Loss given default', 0.0, 1.0, 0.45, 0.01)
            net_yield = st.slider('Net yield for a repaid loan', 0.0, 0.5, 0.10, 0.01)
        failed = economic_cases(frame, exposure, lgd, net_yield)
        st.metric("Cases where the simplified replay disagreed", len(failed))
        missed_risk = int(failed.fixed_decision.eq('Approve').sum())
        lost = int(failed.fixed_decision.eq('Reject').sum())
        st.write(f'Potentially missed risk: {missed_risk} / {len(frame)}; potentially lost opportunity: {lost} / {len(frame)}. Directions are simplified replay decision → full-information decision.')
        if missed_risk == 0:
            st.info('In this dataset, all simplified-replay disagreements are lost-opportunity cases; no risk-underestimation cases were observed. This finding applies only to these 1,000 frozen replays.')
        elif lost == 0:
            st.info('In this dataset, all simplified-replay disagreements are risk-underestimation cases; no lost-opportunity cases were observed. This finding applies only to these 1,000 frozen replays.')
        choice = st.selectbox('Show cases', ['All', 'Missed Risk', 'Lost Opportunity'])
        if choice != 'All':
            failed = failed.loc[failed.fixed_decision.eq('Approve' if choice == 'Missed Risk' else 'Reject')]
        failed = failed.sort_values('Expected Loss Impact Magnitude ($)', ascending=False)
        st.dataframe(failed[["anonymous_applicant_id", "Failure type", "fixed_probability", "full_probability", 'Estimated Expected Loss Impact ($)', 'Expected Loss Impact Magnitude ($)']], width="stretch", hide_index=True)
        with st.expander("Show full financial detail and formula"):
            st.dataframe(failed[["anonymous_applicant_id", 'Fixed expected loss (USD)', 'Full-Evidence expected loss (USD)', 'Full-Evidence expected net contribution (USD)', 'Net contribution change from using Full Evidence (USD)', "fixed_path"]], width="stretch", hide_index=True)
            st.caption('Estimated Expected Loss Impact = (simplified replay risk − full-information risk) × exposure at default × loss given default. A positive value means the simplified replay estimate is higher; a negative value means it is lower. Expected net contribution = exposure × [(1−risk) × net yield − risk × loss given default].')
    with assumptions:
        st.markdown('''**Economic scenario parameters in Failure Explorer:** default EAD (average exposure at default) is **$10,000**, LGD (loss given default) is **45%**, and net yield for a repaid loan is **10%**. These inputs only change the dollar translation; they do not change PDs, policy paths, or any research result.''')
        st.markdown('''**How the frozen path was ordered:** the original study used a stratified 75/25 development split and Logistic Regression validation ROC-AUC to evaluate stepwise backward deletion of complete evidence units. At each round, it selected the removal candidate with the highest retained AUC, using PR-AUC and unit name to break ties, subject to a maximum 0.01 AUC loss relative to Full Evidence. The final order places retained units first and remaining units afterward, preserving original within-group order. This is neither applicant-specific greedy information-gain ordering nor a complete ranking by marginal AUC gain. Source: `vc_credit_eswa_core.py::select_fixed_minimal_units`; B5.1 invokes the same frozen method.

**Why public users cannot score manual inputs or run real-time counterfactuals:** the deployment intentionally publishes only anonymous final replays, frozen paths, and minimal dependencies. It excludes raw application features, feature-engineering rules, per-evidence-state model caches, and training artifacts. Keeping external inputs separate avoids producing PDs that cannot be reproduced from the frozen research environment, and maintains the data-minimisation boundary. These controls remain available only in the local research application. The dollar scenario is arithmetic on published PDs, not a new model counterfactual.

**What the frontier means:** it marks only the empirical non-dominated set among scanned candidates. A development-set frontier does not guarantee test-set optimality. Oracle uses Full-Evidence look-ahead and is diagnostic only, not a deployable candidate.''')
        st.markdown("""- This public demo reads a 1,000-case frozen Home Credit historical replay.
- It does not import research scripts, train a model, tune a threshold, or read TARGET.
- Full Evidence is a model reference, not ground truth.
- Counterfactual simulation and manual-input scoring remain available only in the local research-connected application because their raw evidence-state artifacts are not published.""")

    st.divider()
    story, role, lesson = st.columns(3)
    with story:
        st.subheader("What I tried")
        st.write("Complex dynamic approaches → experiment-led validation → no stable advantage over a simpler rule → deliberate simplification into an explainable replay.")
    with role:
        st.subheader("My role and AI workflow")
        st.write("I owned problem definition, data analysis, experiment design, result interpretation, scope reduction, and prototype integration.")
        st.caption("ChatGPT supported framing and copy exploration; Codex supported code generation, debugging, and test iteration; Gemini supported alternative UI and explanation ideas. I retained responsibility for validation and final product decisions.")
    with lesson:
        st.subheader("Result and lesson")
        st.write("Information reduction ranged from 22% to 61% across frozen experiments, depending on the dataset and model.")
        st.caption("Complexity is not value unless it produces a reliable improvement. The final prototype keeps the smallest explainable mechanism that preserved useful behaviour.")

