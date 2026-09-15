"""Cloud-safe public demo; uses only demo_data frozen artifacts."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from webapp.services.inference_runtime import load_public_runtime


@st.cache_resource
def runtime():
    return load_public_runtime()


def run() -> None:
    st.set_page_config(page_title="Credit Evidence Policy Lab", layout="wide")
    rt = runtime()
    st.title("Credit Evidence Policy Lab")
    st.caption("Public, frozen historical replay. No research experiment runs in this deployment.")
    explore, compare, failures, assumptions = st.tabs(["Explore replay case", "Compare policies", "Failure explorer", "Assumptions"])
    with explore:
        pos = st.selectbox("Historical simulation case", range(len(rt.cases)), format_func=rt.case_id)
        row = rt.cases.iloc[pos]
        a, b, c = st.columns(3)
        a.metric("Fixed-margin PD", f"{row.fixed_probability:.1%}")
        b.metric("Fixed decision", row.fixed_decision)
        c.metric("Full-evidence reference PD", f"{row.full_probability:.1%}")
        st.write(f"**Frozen fixed path:** {row.fixed_path or 'Stopped with initial evidence'}")
        st.write(f"**Full-evidence agreement:** {'Yes' if not row.fixed_disagreement else 'No'}")
        st.caption("This public artifact preserves final frozen replay states. It intentionally does not expose source records or permit new inference / counterfactual recomputation.")
    with compare:
        frame = rt.cases
        results = pd.DataFrame([
            {"Policy": "Full Evidence reference", "Average optional evidence": 5, "Agreement with Full Evidence": 1.0},
            {"Policy": "Fixed Margin", "Average optional evidence": frame.fixed_optional_units.mean(), "Agreement with Full Evidence": 1 - frame.fixed_disagreement.mean()},
            {"Policy": "Experimental Oracle Adaptive", "Average optional evidence": frame.oracle_optional_units.mean(), "Agreement with Full Evidence": 1 - frame.oracle_disagreement.mean()},
        ])
        st.dataframe(results, width="stretch", hide_index=True)
        chart = px.scatter(results, x="Average optional evidence", y="Agreement with Full Evidence", color="Policy", template="plotly_white")
        chart.update_yaxes(tickformat=".1%")
        st.plotly_chart(chart, width="stretch")
        st.caption("Oracle Adaptive is an existing diagnostic that uses Full-Evidence look-ahead; it is not deployable.")
    with failures:
        frame = rt.cases.copy()
        failed = frame.loc[frame.fixed_disagreement].copy()
        failed["Failure type"] = failed.apply(lambda item: "Risk Leakage (Approve → Reject)" if item.fixed_decision == "Approve" else "Lost Opportunity (Reject → Approve)", axis=1)
        st.metric("Frozen Fixed-Margin disagreements", len(failed))
        st.dataframe(failed[["anonymous_applicant_id", "Failure type", "fixed_probability", "full_probability", "fixed_path", "fixed_early_stop"]], width="stretch", hide_index=True)
    with assumptions:
        st.markdown("""- This public demo reads a 1,000-case frozen Home Credit historical replay.
- It does not import research scripts, train a model, tune a threshold, or read TARGET.
- Full Evidence is a model reference, not ground truth.
- Counterfactual simulation and manual-input scoring remain available only in the local research-connected application because their raw evidence-state artifacts are not published.""")
