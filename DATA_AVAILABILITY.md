# 数据来源、许可和复现

本仓库不重新分发个人层面的教育日志、清洗面板或逐人预测。公开不等于可以忽视再识别风险；数据处理仍须遵守来源条件和所在机构要求。

## KU Leuven

来源：[Zenodo 17087849](https://zenodo.org/records/17087849)，DOI `10.5281/zenodo.17087849`。2026-11-16核查该记录API返回的数据许可为`cc-by-4.0`。数据论文：[Tiukhova等，2026，Scientific Data](https://doi.org/10.1038/s41597-026-06821-3)。请同时按数据记录要求引用数据及论文。论文自身的许可与数据记录许可不是同一件事。

[官方去标识化说明](https://pmc.ncbi.nlm.nih.gov/articles/PMC13066476/)说明编号不能用于跨课程、跨学年跟踪原学生。因此本项目不凭裸编号相等认定同一人；也不声称能恢复跨课程学生独立的验证。

```powershell
python paper_output/code/data_processing/download_public_datasets.py --datasets ku_leuven
```

原下载脚本同时登记本地OULAD，若尚无OULAD文件请先完成下一节获取步骤。将`dataset.zip`解压到`crawled_data/raw/ku_leuven/extracted/`，保留压缩包原有结构，再运行：

```powershell
python paper_output/code/modeling/prepare_ku_daily.py
```

输出`paper_output/data_cleaned/ku_daily_panel.parquet`。下载脚本校验来源MD5；本轮使用的本地面板SHA-256见`results/release_manifest.json`。不要用限制行数的开发模式生成投稿结果。

## OULAD

来源：[开放大学官方数据入口](https://research.stem.open.ac.uk/ouanalyse/dataset/)；数据作者[项目说明](https://people.ciirc.cvut.cz/~kuziljak/publication/open_university_learning_analytics_dataset/)列明CC BY 4.0。引用Kuzilek、Hlosta、Zdrahal（2017），DOI `10.1038/sdata.2017.171`。

从官方入口下载原始压缩包，七张CSV表放在`data/`（供原下载登记脚本使用）。生成面板时明确输入目录：

```powershell
python paper_output/code/modeling/prepare_oulad_daily.py --raw-dir data --out-dir paper_output/data_cleaned
```

Q2需要`oulad_daily.parquet`与`oulad_out_of_window_sensitivity.parquet`；完整原始流程还使用数据质量与注销后活动等附表。没有校内私有数据被上传。本轮实证源于公开数据；合成核测试与条件仿真单独标记，不替代外部实证。
