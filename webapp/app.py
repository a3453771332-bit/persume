from __future__ import annotations
import random
import pandas as pd
import streamlit as st
import plotly.express as px
from components.charts import risk_evolution_chart
from webapp.config.evidence_mapping import INITIAL_SOURCE
from webapp.services.research_adapter import counterfactual_evidence, evidence_details, failure_explorer, failure_policy_comparison, failure_timeline, load_engine, nearest_historical_case, operational_portfolio, path_diversity, policy_lab, product_decision, reason_for_state
from webapp.config.operational_policy import PRESETS

st.set_page_config(page_title="Credit Evidence Policy Lab", page_icon="◈", layout="wide")
@st.cache_resource(show_spinner="Loading frozen research inference…")
def get_engine(): return load_engine()

def reset(position, mode, note):
    e = get_engine(); s = e.state(position, 0, st.session_state.margin)
    st.session_state.update(case_position=position, case_mode=mode, case_note=note, mask=0,
        audit=[{"Step": 0, "Action": "Initial assessment", "Evidence source": INITIAL_SOURCE["name"], "PD before": None, "PD after": s["probability"], "Decision before": None, "Decision after": product_decision(s,e.decision_threshold,st.session_state.margin), "Sufficiency": "Sufficient" if s["sufficient"] else "Insufficient", "Reason": reason_for_state(s)}])

def profile(e, pos):
    r=e.raw_test.iloc[pos]; emp=float(r["DAYS_EMPLOYED"]); income=float(r["AMT_INCOME_TOTAL"])/12; credit=float(r["AMT_CREDIT"])
    return {"Age":f"{max(0,round(-float(r['DAYS_BIRTH'])/365.25))} years", "Employment duration":"Not available" if emp>0 else f"{max(0,-emp/365.25):.1f} years", "Monthly income":f"{income:,.0f}", "Requested loan":f"{credit:,.0f}", "Loan-to-income":f"{credit/max(income*12,1):.1f}× annual income"}

def run():
    st.title("Credit Evidence Policy Lab")
    st.caption("Historical credit pre-approval simulation: can the current evidence support a stable decision?")
    e=get_engine()
    with st.sidebar:
        st.header("Policy controls")
        st.session_state.margin=st.slider("Decision-boundary distance (classification margin)",.01,.12,float(st.session_state.get("margin",e.default_margin)),.01)
        st.caption("Stopping uses |PD − decision threshold|; it is not a financial deposit or fee.")
    intake, assessment, policy_tab, operational_tab, failure_tab, assumptions_tab, audit, evaluation=st.tabs(["Explore Applicant","Credit Assessment","Compare Policies","Operational Simulation","Failure Explorer","Assumptions & Limitations","Audit trail","Offline evaluation"])
    with intake:
        mode=st.radio("Intake mode",["Demo case","Manual input simulation"],horizontal=True)
        if mode=="Demo case":
            current=st.session_state.get("case_position",0); c1,c2,c3,c4=st.columns(4)
            if c1.button("Previous case"): current=(current-1)%e.applicant_count
            if c2.button("Next case"): current=(current+1)%e.applicant_count
            if c3.button("Random case"): current=random.randrange(e.applicant_count)
            current=c4.selectbox("Select case",range(e.applicant_count),index=current,format_func=e.applicant_id)
            if st.button("Load demo case",type="primary") or st.session_state.get("case_mode")!=mode: reset(current,mode,"Historical simulation case from the Home Credit dataset.")
        else:
            proto=e.raw_test.iloc[st.session_state.get("case_position",0)]; x,y=st.columns(2)
            with x:
                age=st.number_input("Age",18,90,int(max(18,-proto["DAYS_BIRTH"]/365.25))); years=st.number_input("Employment years",0.,50.,float(max(0,-proto["DAYS_EMPLOYED"]/365.25)),.5)
                income=st.number_input("Monthly income",1.,value=float(proto["AMT_INCOME_TOTAL"]/12),step=100.); credit=st.number_input("Requested loan amount",1.,value=float(proto["AMT_CREDIT"]),step=1000.)
            if st.button("Create simulated application",type="primary"):
                pos=nearest_historical_case(e,{"age":age,"employment_years":years,"monthly_income":income,"loan_amount":credit})
                reset(pos,mode,"Manual input is matched to a nearest held-out historical template; unfilled model fields remain from that template.")
        if "case_position" not in st.session_state: reset(0,"Demo case","Historical simulation case from the Home Credit dataset.")
        p=profile(e,st.session_state.case_position); st.subheader(f"New application · {e.applicant_id(st.session_state.case_position)}"); st.info(st.session_state.case_note)
        st.write(f"**Applicant summary:** {p['Age']} applicant · Employment: {p['Employment duration']} · Monthly income: {p['Monthly income']} · Requested credit: {p['Requested loan']}.")
        for col,keys in zip(st.columns(3),[["Age"],["Employment duration"],["Monthly income","Requested loan","Loan-to-income"]]):
            with col:
                for k in keys: st.write(f"**{k}:** {p[k]}")
    with assessment:
        pos=st.session_state.case_position; s=e.state(pos,st.session_state.mask,st.session_state.margin); decision=product_decision(s,e.decision_threshold,st.session_state.margin)
        st.subheader("Can we decide now?"); a,b,c,d=st.columns(4); a.metric("Predicted default probability",f"{s['probability']:.1%}"); b.metric("Current credit decision",decision); c.metric("Evidence sufficiency","Sufficient" if s["sufficient"] else "Insufficient"); d.metric("Evidence units acquired",f"{len(s['acquired'])}/{len(e.ranking)}")
        st.caption(f"Decision threshold: {e.decision_threshold:.1%} · Boundary distance: {s['margin']:.1%} · {reason_for_state(s)}")
        if s["next_unit"]:
            detail=evidence_details(s["next_unit"]); st.info(f"**Recommended: {detail['name']}** — {detail['description']} It is next in the frozen global policy; current evidence is insufficient.")
            stop_now, gain=st.columns(2)
            with stop_now:
                st.markdown("**Stop now**")
                st.write(f"Decision: {decision}; PD: {s['probability']:.1%}; optional evidence: {len(s['acquired'])}.")
                st.caption("Current state is insufficient. Full-evidence agreement is available only as an offline reference in the Evaluation tab.")
            with gain:
                candidate_mask, _ = e.acquire_next(st.session_state.mask); candidate=e.state(pos,candidate_mask,st.session_state.margin)
                st.markdown("**Acquire recommended evidence**")
                st.write(f"Simulated PD: {s['probability']:.1%} → {candidate['probability']:.1%}; sufficiency: {'Sufficient' if candidate['sufficient'] else 'Insufficient'}.")
                st.caption(f"Illustrative evidence cost: {detail['illustrative_cost']:.0f} points; delay assumption: {detail['delay']}.")
            st.subheader("Counterfactual Evidence Simulator")
            st.caption("Historical counterfactual simulation: results reveal one additional historical evidence group temporarily; simulation does not alter the official path or audit trail.")
            counterfactuals=counterfactual_evidence(e,pos,st.session_state.mask,e.decision_threshold,st.session_state.margin)
            st.dataframe(counterfactuals.drop(columns=["Internal ID"]),width="stretch",hide_index=True,column_config={"PD after":st.column_config.NumberColumn(format="%.1%%"),"Boundary improvement":st.column_config.NumberColumn(format="%.1%%"),"Cost points*":st.column_config.NumberColumn(format="%.0f"),"Delay points*":st.column_config.NumberColumn(format="%.0f")})
            selected_source=st.selectbox("Simulate / acquire evidence source",counterfactuals["Internal ID"].tolist(),format_func=lambda unit:evidence_details(unit)["name"])
            simulated=counterfactuals.loc[counterfactuals["Internal ID"].eq(selected_source)].iloc[0]
            st.write(f"**What-if:** PD {s['probability']:.1%} → {simulated['PD after']:.1%}; boundary change {simulated['Boundary improvement']:+.1%}; would stop: {'Yes' if simulated['Would stop?'] else 'No'}; agreement change: {int(simulated['Agreement change']):+d}.")
            if st.button("Acquire recommended evidence",type="primary"):
                before=s; mask,unit=e.acquire_next(st.session_state.mask); st.session_state.mask=mask; after=e.state(pos,mask,st.session_state.margin)
                st.session_state.audit.append({"Step":len(st.session_state.audit),"Action":"Evidence requested","Evidence source":evidence_details(unit)["name"],"PD before":before["probability"],"PD after":after["probability"],"Decision before":decision,"Decision after":product_decision(after,e.decision_threshold,st.session_state.margin),"Sufficiency":"Sufficient" if after["sufficient"] else "Insufficient","Reason":reason_for_state(after)}); st.rerun()
            if st.button("Acquire selected simulated evidence"):
                before=s; unit=selected_source; mask=st.session_state.mask | (1 << __import__("vc_credit_eswa_core").OPTIONAL_UNITS.index(unit)); st.session_state.mask=mask; after=e.state(pos,mask,st.session_state.margin)
                st.session_state.audit.append({"Step":len(st.session_state.audit),"Action":"Evidence requested after counterfactual simulation","Evidence source":evidence_details(unit)["name"],"PD before":before["probability"],"PD after":after["probability"],"Decision before":decision,"Decision after":product_decision(after,e.decision_threshold,st.session_state.margin),"Sufficiency":"Sufficient" if after["sufficient"] else "Insufficient","Reason":"User-selected historical counterfactual evidence acquired"}); st.rerun()
        else: st.success("Final decision state reached. No further optional evidence is requested.")
        st.plotly_chart(risk_evolution_chart([{ "Step":x["Step"],"PD":x["PD after"]} for x in st.session_state.audit],e.decision_threshold),width="stretch")
    with policy_tab:
        st.subheader("Compare policies across frozen held-out cases")
        size_label=st.selectbox("Replay sample size",["100","500","1000","All held-out cases"],index=0)
        requested=e.applicant_count if size_label=="All held-out cases" else min(int(size_label),e.applicant_count)
        st.caption("Policies make decisions without TARGET. Portfolio bad rate is calculated afterward as a historical offline evaluation metric. *Costs are illustrative assumptions, not observed commercial prices.")
        if st.button("Run policy comparison",type="primary"):
            results, paths=policy_lab(e,requested,st.session_state.margin)
            st.session_state.policy_results,st.session_state.policy_paths,st.session_state.policy_n=results,paths,requested
        if "policy_results" in st.session_state:
            results=st.session_state.policy_results.copy(); st.dataframe(results,width="stretch",hide_index=True,column_config={"Evidence reduction vs full":st.column_config.NumberColumn(format="%.1%%"),"Decision agreement with full":st.column_config.NumberColumn(format="%.1%%"),"Disagreement rate":st.column_config.NumberColumn(format="%.1%%"),"Approval rate":st.column_config.NumberColumn(format="%.1%%"),"Portfolio bad rate (offline)":st.column_config.NumberColumn(format="%.1%%"),"Manual review rate":st.column_config.NumberColumn(format="%.1%%")})
            chart=px.scatter(results,x="Average acquisition cost*",y="Decision agreement with full",color="Policy",size="Average optional evidence units",hover_data=["Disagreement rate","Portfolio bad rate (offline)","Policy source"],template="plotly_white")
            chart.update_yaxes(tickformat=".1%"); chart.update_layout(height=370,margin=dict(l=10,r=10,t=35,b=10)); st.plotly_chart(chart,width="stretch")
            st.caption("Trade-off question: what decision agreement is lost when less evidence is requested? Experimental Oracle Adaptive uses Full-Evidence look-ahead and is not a deployable recommendation.")
            st.subheader("Experimental adaptive-path diversity")
            diversity,examples=path_diversity(st.session_state.policy_paths["Experimental Oracle Adaptive"])
            st.dataframe(diversity,width="stretch",hide_index=True)
            st.write("**Five representative paths:**")
            for path in examples: st.write(f"- {path}")
    with operational_tab:
        st.subheader("Operational Policy portfolio simulation")
        st.caption("Reduce unnecessary evidence collection while making the trade-off in decision reliability explicit.")
        preset=st.selectbox("Business policy preset",list(PRESETS),format_func=lambda name:f"{name} — {PRESETS[name]['description']}")
        default=PRESETS[preset]; u,v=st.columns(2)
        with u:
            op_size=st.selectbox("Simulation sample size",[100,500,1000,"All held-out cases"],index=0)
            op_threshold=st.slider("Decision threshold (scenario parameter)",0.05,0.30,float(e.decision_threshold),0.01)
            op_margin=st.slider("Stopping boundary distance",0.01,0.12,float(default['margin']),0.01)
        with v:
            op_max=st.slider("Maximum optional evidence units",0,len(e.ranking),int(default['max_units']))
            review_penalty=st.slider("Manual review friction penalty (points*)",0.0,4.0,float(default['manual_review_penalty']),0.5)
            st.caption("*Product simulation assumptions. Evidence cost: Low=1, Medium=2, High=3 points; delay points indicate relative operational delay, not actual elapsed time.")
        requested=e.applicant_count if op_size=="All held-out cases" else min(int(op_size),e.applicant_count)
        if st.button("Run operational simulation",type="primary"):
            metrics, queue=operational_portfolio(e,requested,op_threshold,op_margin,op_max,review_penalty)
            trade=[]
            for name, settings in PRESETS.items():
                record,_=operational_portfolio(e,requested,op_threshold,settings['margin'],settings['max_units'],settings['manual_review_penalty']); record['Preset']=name; trade.append(record)
            st.session_state.operational_metrics,st.session_state.review_queue,st.session_state.operational_trade=metrics,queue,pd.DataFrame(trade)
        if "operational_metrics" in st.session_state:
            m=st.session_state.operational_metrics
            st.subheader("Portfolio funnel")
            funnel=pd.DataFrame({"Stage":["Total applications","Decision with initial evidence","Additional evidence required","Manual review required","Automatically approved","Automatically rejected"],"Applications":[m[key] for key in ["Total applications","Decision with initial evidence","Additional evidence required","Manual review required","Automatically approved","Automatically rejected"]]})
            st.plotly_chart(px.funnel(funnel,x="Applications",y="Stage",template="plotly_white"),width="stretch")
            kpi=pd.DataFrame([m]).T.rename(columns={0:"Value"}); st.dataframe(kpi,width="stretch")
            st.subheader("Preset trade-offs")
            trade=st.session_state.operational_trade
            c1=px.scatter(trade,x="Applicant friction index*",y="Decision agreement with full",color="Preset",size="Average optional evidence units",template="plotly_white")
            c1.update_yaxes(tickformat=".1%"); st.plotly_chart(c1,width="stretch")
            c2=px.scatter(trade,x="Average evidence cost points*",y="Approved portfolio bad rate (offline)",color="Preset",size="Manual review rate",template="plotly_white")
            c2.update_yaxes(tickformat=".1%"); st.plotly_chart(c2,width="stretch")
            st.caption("These charts report scenario trade-offs, not a claim that one preset is Pareto-dominant. Bad-rate metrics are historical offline evaluation only.")
            st.subheader("Manual Review Queue")
            queue=st.session_state.review_queue; reviews=queue.loc[queue["Manual review"], ["Case ID","Current PD","Decision threshold","Boundary distance","Evidence acquired","Reason for review","Product decision"]]
            st.dataframe(reviews,width="stretch",hide_index=True,column_config={"Current PD":st.column_config.NumberColumn(format="%.1%%"),"Decision threshold":st.column_config.NumberColumn(format="%.1%%"),"Boundary distance":st.column_config.NumberColumn(format="%.1%%")})
            if len(reviews):
                case=st.selectbox("Open a review case in Explore Applicant",reviews["Case ID"].tolist())
                if st.button("Load selected review case"):
                    pos=next(i for i in range(e.applicant_count) if e.applicant_id(i)==case); reset(pos,"Demo case","Historical simulation case from the Home Credit dataset."); st.success("Case loaded. Open Explore Applicant to inspect its path.")
    with failure_tab:
        st.subheader("Decision Failure Explorer")
        failure_size=st.selectbox("Failure-analysis sample size",[100,500,1000,"All held-out cases"],index=0)
        failure_n=e.applicant_count if failure_size=="All held-out cases" else min(int(failure_size),e.applicant_count)
        if st.button("Analyze policy failures",type="primary"):
            st.session_state.failures=failure_explorer(e,failure_n,st.session_state.margin)
        if "failures" in st.session_state:
            failures=st.session_state.failures; disagree=failures.loc[failures["Failure type"].ne("Agreement")]
            f1,f2,f3,f4=st.columns(4); f1.metric("Total evaluated",len(failures)); f2.metric("Disagreement",len(disagree)); f3.metric("Risk Leakage",int(disagree["Failure type"].str.startswith("Risk Leakage").sum())); f4.metric("Lost Opportunity",int(disagree["Failure type"].str.startswith("Lost Opportunity").sum()))
            st.caption(f"Early-stop failure rate: {failures['Early stop failure'].mean():.1%} · Manual-review potentially preventable share: {disagree['Potentially preventable by review'].mean() if len(disagree) else 0:.1%}. Full Evidence is a model reference, not ground truth.")
            kind=st.multiselect("Failure type filter",["Risk Leakage (Approve → Reject)","Lost Opportunity (Reject → Approve)"],default=["Risk Leakage (Approve → Reject)","Lost Opportunity (Reject → Approve)"])
            table=failures.loc[failures["Failure type"].isin(kind)].drop(columns=["Path","Final mask"]); st.dataframe(table,width="stretch",hide_index=True,column_config={"Current PD":st.column_config.NumberColumn(format="%.1%%"),"Full Evidence PD":st.column_config.NumberColumn(format="%.1%%")})
            if len(table):
                chosen=st.selectbox("Inspect failure case",table["Case ID"].tolist()); st.subheader("Failure timeline")
                st.dataframe(failure_timeline(e,chosen,st.session_state.margin),width="stretch",hide_index=True,column_config={"PD":st.column_config.NumberColumn(format="%.1%%"),"Boundary distance":st.column_config.NumberColumn(format="%.1%%")})
                st.caption("Decision changed after including later evidence in the Full-Evidence reference. This is a historical comparison, not a causal claim.")
                st.subheader("Policy comparison for this case")
                st.dataframe(failure_policy_comparison(e,chosen,st.session_state.margin),width="stretch",hide_index=True)
                st.caption("Risk-Controlled Margin and Experimental Gate are not shown because their frozen experiments are not parameter-compatible with this Web replay. They are not silently reimplemented.")
    with assumptions_tab:
        st.subheader("Business assumptions & limitations")
        st.markdown("""- Home Credit is used only as a historical simulation dataset.
- Evidence requests reveal previously hidden historical feature groups; no external provider is called.
- Evidence cost points and delay points are illustrative operational indexes, not vendor prices or approval-time estimates.
- Full Evidence is a model reference, not ground truth.
- TARGET is used only after decisions for offline evaluation.
- This is a decision-support prototype; regulatory, KYC, fraud, affordability, and adverse-action requirements are not fully modeled.
- Manual Review is a product-layer queue state and never changes the frozen binary model prediction.""")
    with audit:
        st.dataframe(pd.DataFrame(st.session_state.audit),width="stretch",hide_index=True,column_config={"PD before":st.column_config.NumberColumn(format="%.1%%"),"PD after":st.column_config.NumberColumn(format="%.1%%")})
    with evaluation:
        s=e.state(st.session_state.case_position,st.session_state.mask,st.session_state.margin); st.subheader("Reference / evaluation — not part of live decision"); st.write(f"Full-evidence model reference: **{s['full_probability']:.1%}** · {s['full_decision']}")
        with st.expander("Historical observed outcome (offline only)"):
            st.write("Observed default" if int(e.raw_test.iloc[st.session_state.case_position]["TARGET"]) else "Observed no default"); st.caption("Available only because this is a historical offline dataset; never used by the decision workflow.")
