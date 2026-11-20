# agri-worldmodel

第四届 Autonomous Greenhouse Challenge（AGC），路线 A：检验在相同历史和相同作物模型下，RGB-D 相对 RGB 能否改善 14 个当地日历日提前量的矮生番茄总果实鲜重预测。

阶段 0 交付在 [`audit/`](audit/)；先读 [数据审计及复跑说明](audit/data_audit.md)。
`forecast_manifest.csv` 按隔间×共同采样日列出唯一目标，`harvest_gt.csv` 保留单株称重，`compartment_obs.parquet` 仅为 Reference 样表。此仓库不训练模型，也不把未标定的深度原值当作冠层高度。

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m scripts.phase0 --output audit
.venv/bin/python -m unittest discover -s tests -v
```

数据归属：[Maree 等，正式赛 v1](https://doi.org/10.4121/fa102772-32db-4b30-bace-12f2016722ce.v1)、[Righini 等，预试验 v1](https://doi.org/10.4121/e1ee9de9-6ce9-4502-a37c-34b5b1372bed.v1)，4TU.ResearchData，CC BY 4.0。`audit/` 中的派生表和缩略质检图是对来源的筛选、转录或变换。`data/source-cache/` 不提交。
