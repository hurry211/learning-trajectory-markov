# 平台日志不完备条件下的学习轨迹：实验复现与补充验证

这是论文现有工程的独立复现副本，不覆盖原稿。仓库默认私有，仅上传代码、合成测试、模型级参数和汇总结果，不上传学生级日志、编号、逐人预测或文献PDF。

## 先看结论与边界

- HMM优于一阶可观测MC，但新增二阶、三阶MC基线优于当前HMM。因此不能宣称HMM整体最优，也不能由预测优势证明“真实认知状态”。
- KU匿名编号不能跨课程、跨学年关联同一个人。不能按裸`USER_ID`合并学生，也不能用该数据核实真实学生不重叠。可恢复的重采样单位是课程—学年内的注册记录。
- Q3/Q4实际执行的是可观测非齐次MC（NHMC），不是HSMM，也不是成功验证的非齐次HMM。
- 旧Q3模型回退曾参考外层结果；这一部分应称“外层比较与条件情景分析”，不是完全未触碰的外部验证。
- 使用未来已观测行为协变量是回顾性条件情景，不是第28天可部署预测。补充提供冻结地标历史行为的设定，仍不代表因果干预。
- 完成指标是关联模型分数；2021校准误差较大，不能把模拟分数当作真实通过概率或策略效应。

数值、哪些实验已运行及哪些还缺证据，见[补充实验报告](reports/补充实验报告.md)和[证据清单](reports/证据清单.md)。

## 目录

- `experiments/`：新补充实验与精确风险递推。
- `tests/`：不依赖学生数据的边界、递推、配对误差、模型检查。
- `paper_output/code/`：原工程核心代码，保留原目录结构与原算法。
- `paper_output/results/`、`paper_output/tables/`：经白名单筛选的旧模型参数与汇总选模/验证表，作为历史来源，不冒充本轮全部重跑。
- `results/`：本轮实际执行的汇总结果及来源/运行清单。
- `private/`、清洗面板与原始数据：只在本地，Git排除。

## 运行（Python 3.13）

安装依赖：`python -m pip install -r requirements.txt`。建议在E盘创建虚拟环境，并将pip缓存也设置到非C盘。单元测试不需要下载数据：

```powershell
python -m unittest discover -s tests -v
python -m unittest discover -s paper_output/code/modeling/tests -v
```

实证运行前按[数据获取说明](DATA_AVAILABILITY.md)生成清洗面板；先设置数值库线程数，避免多进程启动过量线程：

```powershell
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
New-Item -ItemType Directory -Force private,results | Out-Null
python -m experiments.run_q1_baselines
python -m experiments.run_q1_extension --replicates 10000
python -m experiments.run_q2_extension --replicates 2000
python -m experiments.run_q34_extension --paths 100000 --bootstrap 2000
python tools/build_report.py
python tools/verify_release.py
```

Q2默认重放旧选模记录中的最终K及获胜随机种子，并核对训练似然和下游指标；这不是重新搜索全部网格。确需重跑整个内层网格可加`--refit-grid`，或使用原`run_q2_oulad.py`重新执行全部流程。原七个留课程外折的汇总表保留，但本轮不宣称已将它们全部重新拟合。

当前`private/*.joblib`仅是本机生成的缓存；复跑训练可将其移入另一个本地备份目录。不要加载来源不明的joblib文件。完整重做原始实验可直接运行原脚本的`--help`查看参数。`run_q1_full.py`默认会覆盖它在本复现副本内的同名结果，请先另存本副本结果。

## 统计口径

Q1以相同观测编码、相同42日地标与相同2021后缀计算条件联合似然；高阶MC只用早期学年训练。所有新增阶数均报告，不依测试结果挑选“最好结果”。Q2使用同一学生的两种模型预测差并按学生重采样；AUC/AP差为正更好，Brier差为负更好。Q3/Q4的MCSE只衡量仿真积分误差，不是策略效应或参数不确定性区间；风险DP与MC使用相同核、相同起点及未来窗口事件。

本轮是既有研究的复现、补充诊断和敏感性分析，已用过的外层数据不会因此重新成为“全新测试集”。没有外部认知标签、随机干预或新的锁定校内测试数据，相关效度与因果结论仍未建立。

GitHub Actions只运行无需学生数据的合成实现测试，不自动下载日志或执行实证实验。它的通过不代表论文认知效度或因果结论通过。

代码版权归原作者；本次没有替作者选择开源许可。数据许可与引用要求见数据说明，不能将数据许可当作代码许可。
