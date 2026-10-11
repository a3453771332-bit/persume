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
def runtime():
    return load_public_runtime()


@st.cache_resource
def progressive_replay() -> ProgressiveReplay:
    return load_progressive_replay()


def evidence_name(unit: str) -> str:
    return FRIENDLY_NAMES.get(unit, EVIDENCE_SOURCES.get(unit, {}).get("name", unit.replace("_EVIDENCE", "").replace("_", " ").title()))


def reset_simulation(case_id: str, note: str = "") -> None:
    st.session_state.sim_case_id = case_id
    st.session_state.sim_step = 0
    st.session_state.sim_note = note


def profile_text(profile: pd.Series) -> str:
    return (f"Age {int(profile.age)} · {float(profile.employment_years):.1f} years employed · "
            f"monthly income ${float(profile.monthly_income):,.0f} · requested credit ${float(profile.requested_credit):,.0f} · {profile.housing}")


def confidence_label(distance: float, margin: float) -> str:
    if distance >= max(0.10, margin * 2):
        return "High"
    if distance >= margin:
        return "Medium"
    return "Low"


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
    case_id = st.session_state.sim_case_id
    states = sim.states(case_id)
    step = min(int(st.session_state.sim_step), len(states) - 1)
    current_state = states.iloc[step]
    profile = sim.profile(case_id)
    st.info(st.session_state.get("sim_note") or "Anonymous historical replay. No source application record is shown.")
    applicant, journey, status = st.columns([1.05, 1.25, 1.1])
    with applicant:
        st.markdown("#### Applicant")
        st.caption(str(profile.case_label))
        st.write(f"**Age**  \n{int(profile.age)}")
        st.write(f"**Employment years**  \n{float(profile.employment_years):.1f}")
        st.write(f"**Monthly income**  \n${float(profile.monthly_income):,.0f}")
        st.write(f"**Requested credit**  \n${float(profile.requested_credit):,.0f}")
        st.write(f"**Housing**  \n{profile.housing}")
    with journey:
        st.markdown("#### Evidence journey")
        journey_steps = ["Basic application information", *sim.ranking]
        for index, unit in enumerate(journey_steps):
            name = evidence_name(unit)
            if index < int(current_state.step):
                st.markdown(f"✅ {name}")
            elif index == int(current_state.step):
                st.markdown(f"➡️ **{name} · Current**")
            else:
                st.markdown(f"○ {name}")
    with status:
        st.markdown("#### Decision status")
        st.metric("Risk probability", f"{float(current_state.probability):.1%}")
        st.metric("Current recommendation", str(current_state.decision))
        st.metric("Evidence used", f"{int(current_state.step)} / {len(sim.ranking)}")
        st.metric("Decision confidence", confidence_label(float(current_state.boundary_distance), sim.margin))
        st.metric("Status", "Stop" if bool(current_state.sufficient) else "More evidence needed")
    acquired = [evidence_name(unit) for unit in str(current_state.evidence_acquired).split(";") if unit]
    remaining = [evidence_name(unit) for unit in sim.ranking[int(current_state.step):]]
    st.write(f"**Already used:** {', '.join(acquired) if acquired else 'Basic application information only.'}")
    st.write(f"**Still available:** {', '.join(remaining) if remaining else 'No additional groups.'}")
    if bool(current_state.sufficient):
        st.success("The available information is sufficiently far from the decision boundary, so the simulator stops here.")
        full_decision = str(current_state.full_decision)
        agreement = str(current_state.decision) == full_decision
        a, b, c, d = st.columns(4)
        a.metric("Progressive recommendation", str(current_state.decision))
        b.metric("Full-information reference", full_decision)
        c.metric("Recommendation agreement", "Yes" if agreement else "No")
        d.metric("Evidence saving", f"{(1 - int(current_state.step) / len(sim.ranking)):.0%}")
        a, b, c, d = st.columns(4)
        a.metric("Evidence units used", f"{int(current_state.step)}")
        b.metric("Evidence units saved", f"{len(sim.ranking) - int(current_state.step)}")
        c.metric("Progressive PD", f"{float(current_state.probability):.1%}")
        d.metric("Full-information PD", f"{float(current_state.full_probability):.1%}")
        st.caption("The full-information result is shown only after the simulator stops. It is a model reference, not ground truth.")
    else:
        next_name = evidence_name(str(current_state.next_evidence))
        detail = EVIDENCE_SOURCES.get(str(current_state.next_evidence), {})
        reason = detail.get("description", "It is the next group in the frozen information order.")
        st.warning(f"The current probability is still close to the decision boundary. The simulator recommends requesting **{next_name}** next. {reason}")
        if st.button("Get next information group", type="primary"):
            st.session_state.sim_step = step + 1
            st.session_state.sim_note = f"Added {next_name}."
            st.rerun()
    st.subheader("Decision timeline")
    timeline = states.iloc[:step + 1].copy()
    timeline["Information added"] = timeline["evidence_added"].map(evidence_name)
    timeline["Risk probability"] = timeline["probability"]
    timeline["Risk change"] = timeline["probability"].diff().fillna(0.0)
    timeline["Recommendation"] = timeline["decision"]
    timeline["Next action"] = timeline["sufficient"].map({True: "Stop", False: "Request next group"})
    timeline["Why"] = timeline["sufficient"].map({True: "Information is sufficiently far from the decision boundary.", False: "Current information is still close to the decision boundary."})
    for _, item in timeline.iterrows():
        change = "" if item["Risk change"] == 0 else f" ({item['Risk change']:+.1%})"
        action = "STOP" if item["Next action"] == "Stop" else "Continue"
        st.markdown(f"**Step {item['step']} · {item['Information added']}**  \nRisk {item['Risk probability']:.1%}{change} · {item['Recommendation']}  \n{action} — {item['Why']}")
        if item["step"] < step:
            st.markdown("↓")
    chart = px.line(timeline, x="step", y="probability", markers=True, labels={"step": "Information groups acquired", "probability": "Risk probability"}, template="plotly_white")
    chart.add_hline(y=sim.threshold, line_dash="dash", annotation_text="Decision boundary")
    chart.update_yaxes(tickformat=".1%")
    with st.expander("Risk trajectory"):
        st.plotly_chart(chart, width="stretch")


def render_comparison(rt) -> None:
    st.subheader("Compare policies")
    st.write("Choose the balance between asking for less information and preserving the full-information recommendation.")
    sweep = load_sweep()
    presets = {"Lean · save more information": 0.04, "Balanced": 0.06, "Conservative · preserve more decisions": 0.09}
    selected_name = st.radio("Information-saving priority  ←→  Decision-consistency priority", list(presets), horizontal=True, index=1)
    margin = presets[selected_name]
    point = sweep.loc[sweep.margin.eq(margin)].iloc[0]
    preset_rows = []
    for name, value in presets.items():
        row = sweep.loc[sweep.margin.eq(value)].iloc[0]
        preset_rows.append({"Policy": name, "Agreement": row.agreement, "Extra evidence": row.evidence, "Information saved": 1 - row.evidence / 5})
    st.dataframe(pd.DataFrame(preset_rows), hide_index=True, width="stretch", column_config={"Agreement": st.column_config.NumberColumn(format="%.2%%"), "Extra evidence": st.column_config.NumberColumn(format="%.1f"), "Information saved": st.column_config.NumberColumn(format="%.0%%")})
    a, b, c = st.columns(3)
    a.metric("Selected policy", selected_name.split(" · ")[0])
    b.metric("Agreement with full information", f"{point.agreement:.2%}")
    c.metric("Average extra evidence", f"{point.evidence:.1f}")
    frontier = pareto_points(sweep)
    chart = px.scatter(sweep, x="evidence", y="agreement", hover_data=["margin", "early_stop_failure"], labels={"evidence": "Average extra information groups", "agreement": "Agreement with full-information reference"}, template="plotly_white")
    chart.add_scatter(x=frontier.evidence, y=frontier.agreement, mode="lines+markers", name="Observed trade-off frontier")
    chart.add_scatter(x=[point.evidence], y=[point.agreement], mode="markers", marker={"size": 15, "symbol": "star"}, name="Selected setting")
    chart.update_yaxes(tickformat=".2%", range=[max(0, sweep.agreement.min() - .003), min(1.001, sweep.agreement.max() + .003)])
    st.plotly_chart(chart, width="stretch")
    st.caption("These are precomputed development-set settings, not live policy tuning. The chart joins observed non-dominated points only.")
    frame = rt.cases
    results = pd.DataFrame([
        {"Policy": "Full-information reference", "Average extra information": 5, "Agreement": 1.0},
        {"Policy": "Frozen fixed order", "Average extra information": frame.fixed_optional_units.mean(), "Agreement": 1 - frame.fixed_disagreement.mean()},
        {"Policy": "Experimental oracle diagnostic", "Average extra information": frame.oracle_optional_units.mean(), "Agreement": 1 - frame.oracle_disagreement.mean()},
    ])
    st.dataframe(results, width="stretch", hide_index=True, column_config={"Agreement": st.column_config.NumberColumn(format="%.2%%")})
    st.caption("The oracle diagnostic uses complete-information look-ahead and is included as a research reference, not as a deployable policy.")


def render_failures(rt) -> None:
    st.subheader("When early stopping gets it wrong")
    st.caption("Different mistakes have different business costs. These amounts are illustrative scenario estimates, not realised losses.")
    with st.expander("Business assumptions"):
        exposure = st.number_input("Typical loan exposure (USD)", min_value=0.0, value=10_000.0, step=1_000.0)
        lgd = st.slider("Loss given default", 0.0, 1.0, 0.45, 0.01)
        net_yield = st.slider("Expected loan margin", 0.0, 0.5, 0.10, 0.01)
    failed = economic_cases(rt.cases.copy(), exposure, lgd, net_yield)
    false_approve = failed.loc[failed.fixed_decision.eq("Approve")]
    false_reject = failed.loc[failed.fixed_decision.eq("Reject")]
    a, b = st.columns(2)
    a.metric("False approve · credit-loss exposure", f"${false_approve['Expected Loss Impact Magnitude ($)'].sum():,.0f}")
    a.caption("A risky applicant was approved too early.")
    b.metric("False reject · foregone opportunity", f"${false_reject['Expected Loss Impact Magnitude ($)'].sum():,.0f}")
    b.caption("A potentially good applicant was rejected too early.")
    failed = failed.sort_values("Expected Loss Impact Magnitude ($)", ascending=False)
    columns = ["anonymous_applicant_id", "Failure type", "fixed_probability", "full_probability", "Estimated Expected Loss Impact ($)", "Expected Loss Impact Magnitude ($)", "fixed_path"]
    st.dataframe(failed[columns], width="stretch", hide_index=True, column_config={"fixed_probability": st.column_config.NumberColumn(format="%.1%%"), "full_probability": st.column_config.NumberColumn(format="%.1%%")})
    st.caption("Estimated impact = (early-stop probability − full-information probability) × typical exposure × loss given default. Values are simplified scenario arithmetic, not realised losses.")


def run() -> None:
    st.set_page_config(page_title="Progressive Decision Explorer", layout="wide")
    rt = runtime()
    sim = progressive_replay()
    st.title("Progressive Decision Explorer")
    st.markdown("### Explore how decisions evolve as additional evidence becomes available.")
    st.caption("观察随着信息逐步增加，模型的预测与决策如何变化，以及系统何时判断信息已经足够。")
    st.markdown("`AI-assisted build` &nbsp; `Experiment-led iteration` &nbsp; `Interactive historical replay`")
    st.write("Collecting every document upfront creates delay, applicant friction, privacy exposure, and operational cost. This prototype explores when the system can stop early—and when it should ask for more information.")
    st.info("This prototype replays frozen out-of-sample decision trajectories and does not perform live credit scoring.")
    st.caption("Across frozen experiments, information reduction ranged from 22% to 61%, depending on the dataset and model.")
    simulator, compare, failures = st.tabs(["Explore a decision", "Compare policies", "Understand risk"])
    with simulator:
        render_simulator(sim)
    with compare:
        render_comparison(rt)
    with failures:
        render_failures(rt)
    st.divider()
    st.markdown("**About this prototype**  \nBuilt from frozen out-of-sample experimental results. Designed to explore sequential evidence acquisition rather than perform live credit decisions.  \n（基于冻结的样本外实验结果构建，用于探索渐进式证据获取，不执行实时授信。）")
    st.markdown("[Methodology（方法说明）](https://github.com/a3453771332-bit/persume#readme) · [GitHub](https://github.com/a3453771332-bit/persume)")
    with st.expander("Technical details"):
        st.write("The public app reads frozen prediction states only. It does not import research scripts, train a model, tune a threshold, access TARGET, or call an external evidence provider. Full information is a model reference, not ground truth.")


if __name__ == "__main__":
    run()

