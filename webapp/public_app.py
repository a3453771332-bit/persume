"""Cloud-safe public demo; all decisions are frozen historical replays."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from webapp.config.evidence_mapping import EVIDENCE_SOURCES
from webapp.services.inference_runtime import load_public_runtime
from webapp.services.policy_analysis import economic_cases, load_sweep, pareto_points
from webapp.services.public_simulator import ProgressiveReplay, load_progressive_replay

FRIENDLY_NAMES = {
    "APPLICATION_EXTERNAL_CREDIT_QUALITY_EVIDENCE": "External credit information",
    "REPAYMENT_PERFORMANCE_QUALITY_EVIDENCE": "Repayment history",
    "BUREAU_ACTIVE_CREDIT_CAPACITY_EVIDENCE": "Active credit obligations",
    "BUREAU_CREDIT_HISTORY_QUALITY_EVIDENCE": "Credit history",
    "BUREAU_RECENT_CREDIT_ACTIVITY_EVIDENCE": "Recent credit activity",
    "REPAYMENT_PRIOR_CREDIT_EXPERIENCE_EVIDENCE": "Previous credit applications",
}

@st.cache_resource
def runtime(): return load_public_runtime()

@st.cache_resource
def progressive_replay() -> ProgressiveReplay: return load_progressive_replay()

def evidence_name(unit: str) -> str:
    return FRIENDLY_NAMES.get(unit, EVIDENCE_SOURCES.get(unit, {}).get("name", unit.replace("_EVIDENCE", "").replace("_", " ").title()))

def reset_simulation(case_id: str, note: str = "") -> None:
    st.session_state.sim_case_id, st.session_state.sim_step, st.session_state.sim_note = case_id, 0, note

def profile_text(profile: pd.Series) -> str:
    return (f"Age {int(profile.age)} · {float(profile.employment_years):.1f} years employed · monthly income ${float(profile.monthly_income):,.0f} · requested credit ${float(profile.requested_credit):,.0f} · {profile.housing}")

def render_simulator(sim: ProgressiveReplay) -> None:
    st.subheader("Choose a frozen decision trajectory")
    st.write("Start with basic application information. Request another group only when the current information is not enough for a stable recommendation.")
    if "sim_case_id" not in st.session_state or st.session_state.sim_case_id not in sim.case_ids:
        default_case = sim.templates.loc[sim.templates["case_label"].eq("More Evidence Needed"), "anonymous_applicant_id"].iloc[0]
        reset_simulation(str(default_case), "Loaded a trajectory chosen to show how additional information changes the decision.")
    default = sim.case_ids.index(st.session_state.sim_case_id)
    def case_option(case_id: str) -> str:
        profile = sim.profile(case_id)
        return f"{profile.case_label} · Showcase {case_id.rsplit('_', 1)[1]}"
    selected = st.selectbox("Sample case", sim.case_ids, index=default, format_func=case_option)
    if st.button("Start this case", type="primary"):
        reset_simulation(selected, f"Loaded {case_option(selected)} from the frozen held-out test set.")
        st.rerun()
    case_id, states = st.session_state.sim_case_id, sim.states(st.session_state.sim_case_id)
    step = min(int(st.session_state.sim_step), len(states) - 1)
    current_state, profile = states.iloc[step], sim.profile(case_id)
    st.info(st.session_state.get("sim_note") or "Anonymous historical replay. No source application record is shown.")
    st.caption(f"Case type: **{profile.case_label}** · Anonymous profile: {profile_text(profile)}")
    a, b, c = st.columns(3)
    a.metric("Current risk probability", f"{float(current_state.probability):.1%}")
    b.metric("Current recommendation", str(current_state.decision))
    c.metric("Information used", f"{int(current_state.step)}/{len(sim.ranking)} groups")
    a, b, c = st.columns(3)
    a.metric("Information remaining", f"{len(sim.ranking) - int(current_state.step)} groups")
    b.metric("Information status", "Enough to stop" if bool(current_state.sufficient) else "More information needed")
    c.metric("Next step", "Stop" if bool(current_state.sufficient) else "Get next information")
    acquired = [evidence_name(unit) for unit in str(current_state.evidence_acquired).split(";") if unit]
    remaining = [evidence_name(unit) for unit in sim.ranking[int(current_state.step):]]
    st.write(f"**Already used:** {', '.join(acquired) if acquired else 'Basic application information only.'}")
    st.write(f"**Still available:** {', '.join(remaining) if remaining else 'No additional groups.'}")
    if bool(current_state.sufficient):
        st.success("The available information is sufficiently far from the decision boundary, so the simulator stops here.")
        agreement = str(current_state.decision) == str(current_state.full_decision)
        a, b, c, d = st.columns(4)
        a.metric("Progressive recommendation", str(current_state.decision)); b.metric("Full-information reference", str(current_state.full_decision)); c.metric("Recommendation agreement", "Yes" if agreement else "No"); d.metric("Evidence saving", f"{(1-int(current_state.step)/len(sim.ranking)):.0%}")
        a, b, c, d = st.columns(4)
        a.metric("Evidence units used", f"{int(current_state.step)}"); b.metric("Evidence units saved", f"{len(sim.ranking)-int(current_state.step)}"); c.metric("Progressive PD", f"{float(current_state.probability):.1%}"); d.metric("Full-information PD", f"{float(current_state.full_probability):.1%}")
        st.caption("The full-information result is shown only after the simulator stops. It is a model reference, not ground truth.")
    else:
        next_name = evidence_name(str(current_state.next_evidence))
        reason = EVIDENCE_SOURCES.get(str(current_state.next_evidence), {}).get("description", "It is the next group in the frozen information order.")
        st.warning(f"The current probability is still close to the decision boundary. The simulator recommends requesting **{next_name}** next. {reason}")
        if st.button("Get next information group", type="primary"):
            st.session_state.sim_step, st.session_state.sim_note = step + 1, f"Added {next_name}."
            st.rerun()
    st.subheader("Decision timeline")
    timeline = states.iloc[:step+1].copy()
    timeline["Information added"] = timeline["evidence_added"].map(evidence_name)
    timeline["Risk probability"] = timeline["probability"]
    timeline["Risk change"] = timeline["probability"].diff().fillna(0.0)
    timeline["Recommendation"] = timeline["decision"]
    timeline["Next action"] = timeline["sufficient"].map({True:"Stop",False:"Request next group"})
    timeline["Why"] = timeline["sufficient"].map({True:"Information is sufficiently far from the decision boundary.",False:"Current information is still close to the decision boundary."})
    st.dataframe(timeline[["step","Information added","Risk probability","Risk change","Recommendation","Next action","Why"]].rename(columns={"step":"Step"}), hide_index=True, width="stretch", column_config={"Risk probability":st.column_config.NumberColumn(format="%.1%%"),"Risk change":st.column_config.NumberColumn(format="%+.1%%")})
    chart = px.line(timeline, x="step", y="probability", markers=True, labels={"step":"Information groups acquired","probability":"Risk probability"}, template="plotly_white")
    chart.add_hline(y=sim.threshold, line_dash="dash", annotation_text="Decision boundary"); chart.update_yaxes(tickformat=".1%")
    st.plotly_chart(chart, width="stretch")

def render_comparison(rt) -> None:
    st.subheader("Evidence-use trade-offs")
    sweep = load_sweep(); margin = st.slider("How cautious should the stop rule be?", float(sweep.margin.min()), float(sweep.margin.max()), .05, .01); point=sweep.loc[sweep.margin.eq(margin)].iloc[0]
    a,b=st.columns(2); a.metric("Average extra information groups",f"{point.evidence:.2f}"); b.metric("Agreement with full information",f"{point.agreement:.2%}")
    frontier=pareto_points(sweep); chart=px.scatter(sweep,x="evidence",y="agreement",hover_data=["margin"],labels={"evidence":"Average extra information groups","agreement":"Agreement with full-information reference"},template="plotly_white")
    chart.add_scatter(x=frontier.evidence,y=frontier.agreement,mode="lines+markers",name="Observed trade-off frontier"); chart.add_scatter(x=[point.evidence],y=[point.agreement],mode="markers",marker={"size":15,"symbol":"star"},name="Selected setting"); chart.update_yaxes(tickformat=".2%")
    st.plotly_chart(chart,width="stretch"); st.caption("These are precomputed development-set settings, not live policy tuning. The chart joins observed non-dominated points only.")

def render_failures(rt) -> None:
    st.subheader("Where an early stop changed the reference result")
    exposure=st.number_input("Illustrative average exposure at default (USD)",min_value=0.0,value=10_000.0,step=1_000.0); lgd=st.slider("Illustrative loss given default",0.0,1.0,.45,.01); net_yield=st.slider("Illustrative net yield for a repaid loan",0.0,.5,.10,.01)
    failed=economic_cases(rt.cases.copy(),exposure,lgd,net_yield).sort_values("Expected Loss Impact Magnitude ($)",ascending=False)
    st.dataframe(failed[["anonymous_applicant_id","Failure type","fixed_probability","full_probability","Estimated Expected Loss Impact ($)"]],width="stretch",hide_index=True)
    st.caption("Estimated Expected Loss Impact = (fixed probability − full-information probability) × exposure × LGD. Values are simplified scenario arithmetic, not realised losses.")

def run() -> None:
    st.set_page_config(page_title="Progressive Decision Explorer",layout="wide")
    rt,sim=runtime(),progressive_replay()
    st.title("Progressive Decision Explorer"); st.markdown("### Explore how decisions evolve as additional evidence becomes available."); st.caption("观察随着信息逐步增加，模型的预测与决策如何变化，以及系统何时判断信息已经足够。")
    st.markdown("`AI-assisted build` &nbsp; `Experiment-led iteration` &nbsp; `Interactive historical replay`")
    st.info("This prototype replays frozen out-of-sample decision trajectories and does not perform live credit scoring.")
    st.caption("Across frozen experiments, information reduction ranged from 22% to 61%, depending on the dataset and model.")
    simulator,compare,failures=st.tabs(["Decision explorer","Compare trade-offs","Review impact scenarios"])
    with simulator: render_simulator(sim)
    with compare: render_comparison(rt)
    with failures: render_failures(rt)
    st.divider(); st.subheader("How the project evolved"); st.markdown("**Complex approaches** → **Experimental comparison** → **No stable advantage** → **Simplified stopping rule** → **Interactive prototype**")
    story,role,lesson=st.columns(3)
    with story: st.write("**What I tried**\n\nComplex dynamic approaches → experiment-led validation → no stable advantage over a simpler rule → deliberate simplification into an explainable prototype.")
    with role: st.write("**My role and AI workflow**\n\nI owned problem definition, data analysis, experiment design, result interpretation, scope reduction, and prototype integration.")
    with lesson: st.write("**Result and lesson**\n\nComplexity is not value unless it produces a reliable improvement.")
    with st.expander("Technical details"):
        st.write("The public app reads frozen Home Credit prefix-prediction states. It does not import research scripts, train a model, tune a threshold, access TARGET, or call an external evidence provider. Margin, threshold selection, bootstrap analysis, cross-fitting, fidelity evaluation, and experimental gates remain research details. Full information is a model reference, not ground truth.")

if __name__ == "__main__": run()
