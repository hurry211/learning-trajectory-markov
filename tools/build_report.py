"""Generate aggregate evidence reports and native scientific figures."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def table(frame):
    def cell(v):
        if isinstance(v,(float,np.floating)):
            return f"{v:.6f}"
        return str(v).replace("|","/")
    return "| "+" | ".join(frame.columns)+" |\n|"+"|".join(["---"]*len(frame.columns))+"|\n"+"\n".join(
        "| "+" | ".join(cell(x) for x in row)+" |" for row in frame.itertuples(index=False,name=None))


def main():
    output=ROOT / "reports"
    figures=output / "figures"
    figures.mkdir(parents=True,exist_ok=True)
    q1=pd.read_csv(ROOT / "results/q1_stronger_baselines.csv")
    q2=pd.read_csv(ROOT / "results/q2_paired_differences.csv")
    q2_manifest=json.loads((ROOT / "results/q2_extension_manifest.json").read_text(encoding="utf-8"))
    dp=pd.read_csv(ROOT / "results/q34_dp_mc.csv")
    pair=pd.read_csv(ROOT / "results/q34_paired_mcse.csv")
    prefix=pd.read_csv(ROOT / "results/q34_original_stream_prefix.csv")
    old=pd.read_csv(ROOT / "paper_output/tables/q3/q3_scenario_estimates.csv")
    mapping={"completion":"completion_score_mean","sustained":"sustained_participation_probability","pi_risk":"continuous_pi_risk"}
    errors=[]
    for row in prefix.loc[prefix.context.eq("observed_future_context")].itertuples():
        value=old.loc[old.scenario_id.eq(row.scenario_id),mapping[row.metric]].iloc[0]
        errors.append(abs(row.first_30000_mean-value))
    if max(errors)>1e-12:
        raise AssertionError("original first 30000 paths did not reproduce")
    if q2_manifest["max_abs_brier_difference_from_archived"]>1e-10:
        raise AssertionError("Q2 Brier replay mismatch")
    calibration=pd.read_csv(ROOT / "results/q34_recreated_calibration.csv")
    kernel=pd.read_csv(ROOT / "results/q34_kernel_sensitivity.csv")
    epsilon=pd.read_csv(ROOT / "results/q34_epsilon_sensitivity.csv")
    # Figure 1: paired predictive contrast against increasingly expressive observables.
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"svg.fonttype":"none"})
    fig,ax=plt.subplots(figsize=(7.2,3.6),layout="constrained")
    y=np.arange(len(q1))
    value=q1.hmm_minus_baseline_nat_per_observation.to_numpy()
    ax.errorbar(value,y,xerr=np.vstack([value-q1.ci_low,q1.ci_high-value]),fmt="o",color="#0072B2",capsize=4)
    ax.axvline(0,color=".5",lw=1)
    ax.set(yticks=y,yticklabels=q1.baseline,xlabel="HMM minus baseline: conditional log likelihood (nat/observation)")
    ax.invert_yaxis()
    fig.savefig(figures / "q1_stronger_baselines.svg")
    plt.close(fig)
    # Figure 2: thresholds and covariate information-set sensitivity.
    labels={"reference_observed_profile":"Reference","upper_revisit_profile":"Higher revisit",
        "upper_time_concentration_profile":"Time concentration","joint_supported_upper_profile":"Joint profile"}
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),sharey=True,layout="constrained")
    for ax,context,title in zip(axes,["observed_future_context","frozen_landmark_history"],
        ["Observed future context (retrospective)","Frozen landmark history (assumed)"]):
        for key,label in labels.items():
            group=dp.loc[dp.context.eq(context)&dp.scenario_id.eq(key)]
            ax.plot(group.run_length,group.risk_dp_equal_anchor,marker="o",label=label)
        ax.set(title=title,xlabel="Consecutive future inactive days (threshold)",xticks=[3,5,7,10,14])
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Exact conditional event risk")
    axes[1].legend(fontsize=8)
    fig.savefig(figures / "q34_risk_thresholds.svg")
    plt.close(fig)
    # Figure 3: paired AUC differences; all landmarks/outcomes retained.
    auc=q2.loc[q2.metric.eq("AUC")].reset_index(drop=True)
    fig,ax=plt.subplots(figsize=(7.2,3.8),layout="constrained")
    value=auc.hmm_minus_observable.to_numpy()
    ax.errorbar(value,np.arange(len(auc)),xerr=np.vstack([value-auc.ci_low,auc.ci_high-value]),
        fmt="o",color="#009E73",capsize=4)
    ax.axvline(0,color=".5",lw=1)
    ax.set(yticks=np.arange(len(auc)),yticklabels=[f"Day {r.landmark}: "+("adverse final" if r.target=="adverse_final" else "next-28d withdrawal") for r in auc.itertuples()],
        xlabel="Paired AUC difference: +HMM minus observable history")
    ax.invert_yaxis()
    fig.savefig(figures / "q2_paired_auc.svg")
    plt.close(fig)
    select_q2=q2.loc[q2.landmark.eq(28),["target","metric","hmm_minus_observable","ci_low","ci_high"]]
    risk7=dp.loc[dp.run_length.eq(7),["context","scenario_id","risk_dp_equal_anchor","risk_mc","conditional_mcse"]]
    eps_select=epsilon.loc[epsilon.epsilon_completion.eq(.005)&epsilon.epsilon_risk.eq(.005),
        ["context","scenario_id","point_nondominated","registration_cluster_nondominated_frequency"]]
    report=f"""# 补充实验报告

本轮补充于2026-11-16至17日执行。原稿与原始结果未被覆盖；本轮使用公开KU/OULAD清洗面板。当前仓库是复现、补充基线与诊断，不是新的未触碰测试集。

## 1. 最重要的研究结论修正

1. HMM相对一阶MC有预测增益，但二阶、三阶可观测MC更强。结果支持“平台序列具有超出一阶可观测链的记忆”，不支持“HMM最优”或“隐状态是真实认知状态”。保留所有阶数，不能只报告有利比较。
2. Q2第28日的两项AUC提升，其配对95%区间均跨0；不能把点估计提升说成已证实的稳定提升。部分Brier/AP对比的区间不跨0，但属于多项探索比较，未做多重检验校正，也未覆盖模型选择不确定性。
3. KU匿名编号不能跨课程、学年关联同一人。撤回按裸编号聚类的中间诊断；不把数字碰撞当作真实学生重叠。主结果以注册记录为可恢复单位，未知跨课程相关性是限制。
4. Q3原回退参考过外层比较；回顾性未来行为路径也不是实时可用信息。冻结历史设定是敏感性，不会把观察模型变成因果干预模型。
5. 通过关联分数在2021存在明显校准漂移。高分不等于可靠通过概率；策略优化必须限于假设模型中的条件情景与权衡，不能给出有效干预的承诺。

## 2. Q1：固定模型重放与更强基线

固定原HMM参数，与原1399条2021序列的条件似然逐条核对，最大绝对误差约{q1.fixed_hmm_replay_max_abs_loglik_error.max():.2e}；一阶MC参数最大误差为{q1.mc1_parameter_replay_max_abs_error.max():.2e}。各模型均使用相同观测编码、42日历史前缀和后续评价窗口；新增可观测链仅用1819+1920训练，Dirichlet平滑0.5。区间为5000次注册级配对重采样，条件于已拟合参数。

{table(q1[["baseline","hmm_minus_baseline_nat_per_observation","ci_low","ci_high"]])}

正数有利于HMM，负数有利于该可观测基线。新增阶数是事后稳健性诊断，不能宣称在全新保留集上完成了预注册选模。

![更强基线](figures/q1_stronger_baselines.svg)

## 3. Q2：OULAD配对增量与地标敏感性

使用旧内层选模记录确定的K=5、最终随机种子436572576，重放最终训练；没有重新挑选种子，训练似然与旧值相同。原始全网格、七个留模块外折的汇总表保留为旧实验来源，不宣称本轮全部重跑。重新生成14/28/56日地标下、两项结局的逐人预测，文件仅保留在本地`private/`。按学生进行2000次配对cluster bootstrap，比较AUC、AP和Brier。所有12组下游模型的AUC与旧表最大差为{q2_manifest['max_abs_auc_difference_from_archived']:.2e}，Brier最大差为{q2_manifest['max_abs_brier_difference_from_archived']:.2e}。28/56日的AUC与旧表相同；14日存在微小排名指标差异，不能宣称全部逐位相同。该差异可能与浮点排序或并列预测分数有关，但没有旧逐人预测供逐对核实；保留差异，不调参追齐。

第28日主分析：

{table(select_q2)}

AUC/AP差为正较好，Brier差为负较好。固定拟合模型的测试集重采样并不包含重训练、状态数选择、超参数搜索的不确定性。完整三地标结果均在`q2_paired_differences.csv`。

![OULAD配对AUC](figures/q2_paired_auc.svg)

## 4. Q3/Q4：数学公式、仿真及精度

重拟合旧NHMC；旧种子生成的前30000条共同随机数路径，12个情景—指标均值与旧表最大差为{max(errors):.2e}。随后每情景、每协变量设定扩大至100000路径，共4情景×2设定=800000条路径；共同随机数用于配对比较。

风险事件是未来28日内出现完全位于该窗口内的连续m个Platform-Inactive日，历史连续零计数重置为0。对m=3、5、7、10、14、两种设定、四情景，共40组同核同起点的DP—MC比较。全部误差在5倍条件MC标准误以内，最大标准化误差{dp.absolute_standardized_error.max():.3f}；这是数值一致性诊断，不是教育效度检验。比较采用实际抽到的起点条件DP，避免把起点混合抽样误差误认为递推错误。

m=7的结果：

{table(risk7)}

![风险阈值与信息集敏感性](figures/q34_risk_thresholds.svg)

配对误差使用D=指标(情景)-指标(参照)，MCSE=sd(D)/sqrt(N)。它只衡量数值积分误差，不是参数区间或因果效应区间。完整结果在`q34_paired_mcse.csv`。

## 5. 稳健性与多目标权衡

epsilon_completion与epsilon_risk各取0、0.001、0.002、0.005、0.01、0.02，交叉形成36个阈值组合；两种协变量设定均运行。每个点直接与全部原始候选比较，不用可能受非传递关系影响的顺序剪枝。用2000次注册级重采样估计前沿稳定性，固定核与校准模型。

epsilon两项均为0.005时：

{table(eps_select)}

另外，对每个情景、每个设定，把转移核分别向均匀核与全Platform-Inactive核混合0.5%、2%、5%，共{len(kernel)}项压力测试；路径事件的逐起点概率差均未超过耦合TV上界，违规数为{int(kernel.bound_violations.sum())}。这些人为扰动不是拟合参数置信域，也不是可识别的学生策略效应。

## 6. 校准和依然缺失的证据

{table(calibration[["partition","pass_rate","mean_predicted_probability","expected_calibration_error_10_bins"]])}

旧标签`untouched_outer_2021`只表示旧校准脚本的分区名；本轮复查不是全新的未触碰验证。没有重调该校准模型来改善外层数值。

尚不能由现有公开日志补出的证据：校内锁定测试集、认知状态外部标签、策略随机干预、真实跨课程学生链接、HSMM显式时长拟合。七折外部验证本轮仅保留旧聚合记录。投稿前应按这些边界修改摘要、结论和局限性；不能用更多仿真替代缺失的实证。

## 7. 对论文写作的直接建议

将主线表述为“平台行为轨迹的记忆结构、隐状态模型的预测边界，以及假设条件下的风险—分数权衡”。数学部分证明递推正确性和扰动稳定性；实证部分允许可观测模型胜出。将二阶、三阶MC放入正式基线表，弱化HMM方法优越性；将Q3/Q4称为条件情景评估而非已验证策略优化。不要使用深度思考、偷懒等没有外部标签支持的状态命名。
"""
    (output / "补充实验报告.md").write_text(report,encoding="utf-8")
    evidence="""# 证据清单

| 项目 | 本轮状态 | 结果/限制 |
|---|---|---|
| Q1固定HMM与一阶MC重放 | 已运行并逐序列/参数核对 | 与旧值数值一致 |
| IID、二阶、三阶MC强基线 | 已训练、评价、配对bootstrap | 二阶/三阶优于当前HMM，必须保留 |
| KU编号作用域 | 已核对官方去标识化说明 | 不能跨课程/学年识别同一学生 |
| Q2最终模型重放 | 已运行 | 旧K及获胜seed，不是新全网格搜索 |
| Q2三地标×两结局配对CI | 已运行，2000次/组 | AUC/AP/Brier，固定模型学生聚类 |
| Q2全内层网格与七留课程外折 | 保留旧汇总和原代码 | 本轮未完整重新拟合；中断过的搜索不计为完成 |
| Q3旧30000路径重放 | 已运行并比较12个旧均值 | 完全一致 |
| 两种信息集×四情景仿真 | 已运行，每组100000路径 | 回顾性条件路径与冻结历史假设 |
| 同核DP—MC风险核对 | 已运行，40组 | 所有差值位于5 MCSE内 |
| 配对路径MCSE | 已运行 | 数值误差，不是因果区间 |
| epsilon二维网格与前沿稳定性 | 已运行，36组合×2信息集，2000次注册bootstrap | 直接全候选比较 |
| 转移核扰动与TV界 | 已运行，48项压力测试 | 人为扰动，不是参数置信域 |
| 合成恢复/边界测试 | 已运行 | 仅证明实现对指定生成机制的行为 |
| HSMM | 未实现，也未报告HSMM结果 | HMM时长PPC不能冒充HSMM |
| 认知标签外部效度 | 无此数据 | 只用编号/平台行为语义 |
| 学习策略因果效果 | 未建立 | 需要识别设计或前瞻随机试验 |
| 校内完全锁定外部测试 | 等待数据 | 不能用已查阅公开测试集冒充 |

本轮没有删除不利结果、按外层表现重新挑选参数，或把数值自检当作论文全部证据通过。详细来源和输入/代码哈希见`results/release_manifest.json`。
"""
    (output / "证据清单.md").write_text(evidence,encoding="utf-8")
    # Include-able LaTeX (not a second manuscript); no fabricated compilation claim.
    lines=[r"% Generated aggregate supplement; include inside an existing Chinese LaTeX document.",
        r"\section{补充实验：预测边界与数值一致性}",
        r"所有新增检验均为既有数据上的复现和事后诊断，不构成新的未触碰测试。",
        r"\subsection{更强可观测基线}",
        r"同一观测编码和42日地标下，HMM与各基线条件对数似然差如下（nat/观测）。正数有利于HMM。",
        r"\begin{center}\begin{tabular}{lrrr}\hline 基线 & 差值 & 95\%下界 & 95\%上界\\\hline"]
    for row in q1.itertuples():
        lines.append(f"{row.baseline} & {row.hmm_minus_baseline_nat_per_observation:.6f} & {row.ci_low:.6f} & {row.ci_high:.6f} " + r"\\")
    lines += [r"\hline\end{tabular}\end{center}",
        r"二阶与三阶可观测链优于当前HMM，因此本文不主张隐状态模型具有普遍预测优势。",
        r"\subsection{OULAD配对区间}",
        r"按学生进行2000次配对重采样。第28日两类结局的AUC增益区间均跨0；Brier差为负表示改善。该区间条件于已拟合模型，不包含选模不确定性。",
        r"\begin{center}\begin{tabular}{llrrr}\hline 结局 & 指标 & 差值 & 下界 & 上界\\\hline"]
    for row in select_q2.itertuples():
        target="终局不利结果" if row.target=="adverse_final" else "未来28日注销"
        metric="AP" if row.metric=="average_precision" else row.metric
        lines.append(f"{target} & {metric} & {row.hmm_minus_observable:.6f} & {row.ci_low:.6f} & {row.ci_high:.6f} "+r"\\")
    lines += [r"\hline\end{tabular}\end{center}",r"\subsection{风险递推与条件情景}",
        f"在两个协变量设定、四个情景和五个连续无活动阈值下，共40组DP与MC比较，最大标准化数值差为{dp.absolute_standardized_error.max():.3f}。每情景仿真100000路径。",
        r"采用共同随机数，以配对差$D$的$\operatorname{sd}(D)/\sqrt{N}$报告仿真误差，不将其解释为策略效应区间。",
        r"未来28日风险只计入完全处于未来窗口内的连续无活动事件；地标处历史连续计数重置。",
        r"原未来已观测行为路径属于回顾性条件情景；冻结地标历史是另一假设设定，二者均不是可识别的因果干预。",
        r"KU编号不能跨课程、学年关联同一学生，故无法验证真实学生不重叠。完成分数在2021存在明显校准漂移，不应被解释为可靠通过概率。"]
    (output / "supplement_experiments.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (ROOT / "results/reproduction_checks.json").write_text(json.dumps({
        "q34_original_30000_path_aggregate_max_abs_difference":float(max(errors)),
        "q34_same_kernel_dp_mc_checks":len(dp),"q34_max_standardized_error":float(dp.absolute_standardized_error.max()),
        "q34_kernel_stress_checks":len(kernel),"q34_epsilon_rows":len(epsilon),
        "q2_max_auc_replay_error":q2_manifest["max_abs_auc_difference_from_archived"],
        "q2_max_brier_replay_error":q2_manifest["max_abs_brier_difference_from_archived"]},indent=2),encoding="utf-8")
    print("Aggregate report, evidence ledger, LaTeX supplement and 3 SVG figures generated.")


if __name__=="__main__":
    main()
