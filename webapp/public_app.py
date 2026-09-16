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
        st.subheader('Margin 参数扫描：开发集证据—一致性权衡')
        sweep = load_sweep()
        margin = st.select_slider('决策边界距离 margin（非保证金、非概率置信区间）', options=sweep.margin.tolist(), value=0.05)
        point = sweep.loc[sweep.margin.eq(margin)].iloc[0]
        left, right = st.columns(2)
        left.metric('开发集平均可选证据量', f'{point.evidence:.3f}')
        right.metric('开发集全证据一致性', f'{point.agreement:.3%}')
        frontier = pareto_points(sweep)
        chart = px.scatter(sweep, x='evidence', y='agreement', hover_data=['margin', 'early_stop_failure'], labels={'evidence': '平均可选证据量', 'agreement': '与全证据决策的一致性'}, template='plotly_white')
        chart.add_scatter(x=frontier.evidence, y=frontier.agreement, mode='lines+markers', name='已扫描点中的经验帕托前沿')
        chart.add_scatter(x=[point.evidence], y=[point.agreement], mode='markers', marker={'size': 15, 'symbol': 'star'}, name='所选 margin')
        chart.update_yaxes(tickformat='.2%', range=[max(0, sweep.agreement.min()-.003), 1.001])
        st.plotly_chart(chart, width='stretch')
        increments = frontier[['margin', 'evidence', 'agreement']].copy()
        increments['每增加一份证据的一致性提升（百分点）'] = 100*increments.agreement.diff()/increments.evidence.diff()
        st.dataframe(increments, hide_index=True)
        st.caption('来自冻结开发集 policy-tune 扫描；连线仅连接已有非支配点，不代表中间阈值已评估。每项证据暂按等成本计。该图不是测试集性能，也不用于重新选择论文策略。滑块查阅已有计算，不重新推理。')
        st.subheader('独立展示：1,000 条冻结测试回放')
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
        exposure = st.number_input('假设平均授信敞口 EAD（USD）', min_value=0.0, value=10000.0, step=1000.0)
        lgd = st.slider('假设 LGD（违约损失率）', 0.0, 1.0, 0.45, 0.01)
        net_yield = st.slider('假设同周期正常还款净收益率（扣除资金及运营成本）', 0.0, 0.5, 0.10, 0.01)
        failed = economic_cases(frame, exposure, lgd, net_yield)
        st.metric("Frozen Fixed-Margin disagreements", len(failed))
        leakage = int(failed.fixed_decision.eq('Approve').sum())
        lost = int(failed.fixed_decision.eq('Reject').sum())
        st.write(f'错过风险：{leakage} / {len(frame)}；失去机会：{lost} / {len(frame)}。方向均为「局部决策 → 全证据决策」。')
        if leakage == 0 or lost == 0:
            st.info('当前冻结子样本只观察到单向失败（或无失败）；零计数不是功能缺失，也不能据此认定总体策略具有单向偏差。')
        choice = st.selectbox('失败方向', ['全部', '错过风险', '失去机会'])
        if choice != '全部':
            failed = failed.loc[failed.fixed_decision.eq('Approve' if choice == '错过风险' else 'Reject')]
        st.dataframe(failed[["anonymous_applicant_id", "Failure type", "fixed_probability", "full_probability", '局部预期损失 USD', '全证据预期损失 USD', '损失估计修正 USD', '全证据预期净贡献 USD', '改按全证据决策的净贡献变化 USD', "fixed_path"]], width="stretch", hide_index=True)
        st.caption('情景假设：预期损失 = PD × LGD × EAD；预期净贡献 = EAD × [(1−PD) × 净收益率 − PD × LGD]；拒绝的净贡献设为 0。PD 与收益率假设同周期，不计提前还款、折现等。金额不是已实现损失或收入；全证据 PD 也只是模型参考。失去机会不必然意味着正利润，负值原样保留。')
    with assumptions:
        st.markdown('''**冻结路径如何排序**：原研究在开发数据上做 75/25 分层划分，用 LR 验证 ROC-AUC 评估完整证据单元的逐步后向删除；每轮选择删除后 AUC 最高的候选，以 PR-AUC 和单元名打破平局，约束相对全证据 AUC 损失不超过 0.01。最终排序将保留单元放在前面、其余单元放在后面，各组沿用原定义顺序。它不是逐申请人的贪心信息增益排序，也不是按边际 AUC 增益完全排序。来源：vc_credit_eswa_core.py::select_fixed_minimal_units；B5.1 调用同一冻结方法。

**为何没有实时反事实评分**：公开包只发布最终预测与路径，未发布原始证据、状态模型及所有中间预测，因此无法可靠重算修改输入后的 PD。这是当前发布范围和运行依赖的选择；没有证据表明存在专门的法律禁令。金额情景仅对已有 PD 作算术换算，不构成新的模型反事实。

**前沿的含义**：这里只标记已扫描候选点中的经验非支配集合；开发集前沿不保证在测试集保持最优。Oracle 使用全证据前瞻，只用于诊断，不属于可部署候选策略。''')
        st.markdown("""- This public demo reads a 1,000-case frozen Home Credit historical replay.
- It does not import research scripts, train a model, tune a threshold, or read TARGET.
- Full Evidence is a model reference, not ground truth.
- Counterfactual simulation and manual-input scoring remain available only in the local research-connected application because their raw evidence-state artifacts are not published.""")
