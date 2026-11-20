"""Run the AGC route A phase-0 audit from pinned archives."""

import argparse
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import pandas as pd

from .imaging import DEMO_COMMIT, canopy, pretrial_workbooks, reference, single_plant
from .sources import Archive, DOIS, SOURCES, ROOT, small
from .trial import DAYS, ENV, PREFIX, TEAMS, density_events, final_dates, gap_audit, harvest, load_series, manifest, pairs


def table(headers, rows):
    def fmt(i):
        if pd.isna(i):
            return '—'
        if isinstance(i,float):
            return f'{i:.3f}'
        return str(i).replace('|',r'\|').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |',
                      *('| '+' | '.join(fmt(i) for i in row)+' |' for row in rows)])


def report(out, source_rows, gt, stat, ledger, events, obs, forecast, stats, gaps, dst, series,
           single, can, can_index, manual, crop, failures):
    n_common = int(gt.harvest_kind.eq('common').sum()) if not gt.empty else 0
    n_final = len(gt)-n_common
    exp = len(forecast)
    eligible = int(forecast.index_eligible.sum())
    ready = int(forecast.rgbd_model_ready.sum())
    text = f'''# AGC 第四届｜路线 A 阶段 0 数据审计

**判定：清单级验收 {eligible}/{exp}，决定性 RGB-D 比较就绪 {ready}/{exp}；暂不开始模型训练。**
目标是同一机理模型、同一历史帧、同一同化流程下，用独立破坏性称重检验深度是否改善每株总果实鲜重预测。本轮不估计效果。

## 来源与完整性

- 2024 正式赛：[4TU v1，Maree 等](https://doi.org/10.4121/{DOIS[24]}.v1)；2023 预试验：[4TU v1，Righini 等](https://doi.org/10.4121/{DOIS[23]}.v1)。派生数据基于原始数据集，CC BY 4.0。原始归档不入 Git。
- [WUR 点云示例](https://github.com/WURgreenhouse/point-cloud-demo/tree/{DEMO_COMMIT})；[正式赛方法与结果](https://doi.org/10.3390/s25144321)，特别是 §2.4.1、§3.3.2。
- 小文件全量下载，核对官方 MD5 并计算 SHA-256（`source_inventory.csv`）；图像归档只读取 1 MiB 尾部 ZIP64 目录和选中成员的字节范围，要求 HTTP 206 及匹配的 Content-Range，不计算大包整包 MD5。`verified_members.csv` 记录已读取原始成员的 CRC32 和 SHA-256。目录级配对不等于所有 PNG 都已解码。

{table(['文件','字节','发布方 MD5','验证方式'], ((r['filename'],r['bytes'],r['published_md5'],r['verification']) for r in source_rows))}

## 1. 称重真值

- `harvest_gt.csv`：{n_common} 条共同称重单株记录，{n_final} 条终采单株记录。工作簿 `Information!B12:D12` 把 FW 定义为克，实测单位为 g/株；首轮用 `FW `，其余用 `FW tot`（部分是单株颜色分项之和的原公式）。复用原表缓存值，不推算缺失总重、不插值。标准差采用样本标准差（ddof=1）。同名测量株号可跨破坏性日期重用，不是纵向单株 ID。
- `harvest_row_audit.csv` 逐行保留非空源记录及原始单元格、原公式、数据/汇总分类；`harvest_summary.csv` 列出日期来源、样本数、均值、SD 和行号。`Information!B5` 文案写三个隔间，实际三个日期各六个。终采表未给采样日，使用各队唯一 `dwarf_tomato/harvest_date` 日志的当地日期（数值核对为年积日），非实测时刻。终采不计入主评估。
'''
    if not ledger.empty:
        issue = ledger.loc[ledger.quality_flags.ne('') & ledger.row_kind.ne('individual_missing_weight')]
        counts = ledger.loc[ledger.harvest_kind.eq('final') & ledger.row_kind.eq('individual_missing_weight')].groupby('compartment').size()
        text += f'\n- 终采总鲜重缺失的带编号行：{", ".join(f"{c}={n}" for c,n in counts.items())}；有果数或空白不等于有称重，不作为 0。\n'
        text += '\n' + table(['源表','行','株号/原值','总 FW（原值）','异常/处理'],
                              ((r.source_sheet,r.source_row,r.raw_plant,r.raw_total_fresh_weight,r.quality_flags) for r in issue.itertuples())) + '\n'
        outlier = gt.loc[gt.harvest_kind.eq('common') & gt.compartment.eq('3.08') & gt.target_date.eq(DAYS[1])]
        if outlier.quality_flags.str.contains('low_mass').any():
            sensitivity = outlier.loc[~outlier.quality_flags.str.contains('low_mass')].fruit_fresh_weight_g_per_plant
            text += f'\n`308-36` 的 19.62 g/61 果无法确认录入错误，主表保留；不使用该株的敏感性目标为 {sensitivity.mean():.2f} g/株（n={len(sensitivity)}），原目标为 {outlier.fruit_fresh_weight_g_per_plant.mean():.2f}。10 月 22 日第 43 行汇总写 307、公式取 308 数据，不纳入真值。\n'
    if not stat.empty:
        text += '\n' + table(['隔间','日期','类型','n','均值 g/株','SD','源行'],
                             stat[['compartment','target_date','harvest_kind','n','mean_g_per_plant','sd_g_per_plant','source_rows']].itertuples(index=False,name=None)) + '\n'
    text += '''
## 2. 时间、配对与密度

映射核对来源：归档内 `dataset4tu/ReadMe.md`，Reference 为 3.06、`cam_1`。仅按相机 ID 与同一个完整当地文件名时间戳做一对一联结；不用最近邻补模态。`frame_index.csv` 保留源成员、当地时间、UTC 与重复/缺配状态。下表以每台相机首次至末次的 04/08/12/16 时槽为范围，首尾范围外不补；明细见 `frame_gaps.csv`。
'''
    text += '\n' + table(['隔间','相机','首末日（首末小时）','RGB','Depth','完成对','未配对','内部时槽异常'],
                         ((r['compartment'],r['camera'],f"{r['first'][0]} {r['first'][1]:02}:xx → {r['last'][0]} {r['last'][1]:02}:xx",
                           r['rgb'],r['depth'],r['paired'],r['unpaired'],r['issues']) for r in stats)) + '\n'
    if not gaps.empty:
        text += '\n' + table(['隔间','日期','小时','异常'],gaps[['compartment','local_date','hour','issue']].itertuples(index=False,name=None)) + '\n'
    text += '''
影像按 `Europe/Amsterdam` 解释后转 UTC；无法唯一定位的本地时间排除。CSV 保留原偏移并核对 Amsterdam，不二次本地化。下表用 Reference 2024-10-26/27 影像验证秋季切换（CSV 时间以 UTC 排序，不把重复的 02:xx 本地钟时当作重复绝对时间）。
'''
    text += '\n' + table(['当地影像时刻','UTC'],dst)
    if 'reference' in series:
        r = series['reference']
        n_dst = int(r.local.dt.strftime('%Y-%m-%d %H').eq('2024-10-27 02').sum())
        text += f'\nReference CSV 在 2024-10-27 重复本地钟时 02:xx 共 {n_dst} 行，UTC 均唯一。\n'
    text += '''
`density_events.csv` 保存日志时间及原始 CSV 行号。只用不晚于查询时刻的最后一个有效事件（株/m²），不反向填充、不对事件插值。日志多为午夜，不作为物理搬盆时刻。论文 §3.3.2 明确 IDEAS 最后一次由工作人员于 2024-10-20（周日）执行，对应 10-21 的 42→30 日志；无可核实时刻。其余具体搬盆时刻待核实，也未把拍摄变化当作搬盆时刻。
'''
    if not events.empty:
        text += '\n' + table(['隔间','密度事件（当地日：株/m²）'],
            ((comp,'；'.join(f'{r.log_date_local}: {r.plants_m2:g}' for r in g.itertuples())) for comp,g in events.groupby('compartment'))) + '\n'
    text += '''
## 3. Reference 样表

`compartment_obs.parquet` 是诊断样表，不是训练特征。三次取各 17:00 截点之前最近的配对；`reference_cutoff_join.csv` 逐截点连接这些视觉历史样本、当时密度和后续独立称重（后者仅为目标）。另取 10-07 密度事件两侧的 16 时段帧和早期参考平面检查帧。10-06/07 两帧只围绕日志事件，实际搬盆时刻未知；也不能将这两帧加入早于它们的截点。

RGB 使用全图 ROI（`x0,y0,x1,y1`，上界不含）及固定 RGB 色差筛查掩码，Depth 不参与分割。同一掩码用于 RGB 像素覆盖率和掩码内非零且非 65535 的深度比例（编码层有效，不等同于测距准确）。`reference_mask_qc.jpg` 每行左原图、右掩码保留像素，按样表成功解码行排序；掩码未经人工标注验证，存在叶片漏分与背景误分，不用于阈值标定。掩码阈值仅为阶段 0 诊断，不能从这个未来可能留出的隔间继承为拟合阈值。

归档声明 RGB 与 Depth 对齐；PNG 样本验证同一像素网格且 Depth 为 uint16。`reference_alignment_qc.jpg` 并排记录早期一对 RGB 与按百分位拉伸的深度（黑色为无效），用于检查空间结构是否大体对应，不是标定靶残差验证。`cam_1.json` 的 `color_int`（含焦距和主点）及内容哈希随行留档。配置无显式 `DepthScale`，WUR 程序缺省为 0.001 m/单位，原始数值量级可作合理性检查，却未做实物尺度标定。早期帧可见盆和台面，但约 1.5 m 的安装距离不是实测盆顶平面；没有可重复定位的盆顶基准点、平面法向和跨日期稳定性证据。**H90 保留空值及原因**；后续应在经验证的盆顶基准下，将同一 RGB 掩码内有效点反投影为三维坐标，取距该平面朝冠层方向的有符号距离 P90，不用最大深度或无量纲差值代替。
'''
    if not obs.empty and 'depth_raw_median' in obs:
        text += '\n' + table(['样本角色','当地时间','株/m²','RGB 覆盖','掩码有效深度','非零有效深度中位原值','H90 m'],
                             obs[['role','timestamp_local','plants_m2','rgb_coverage_fraction','mask_valid_depth_fraction','depth_raw_median','h90_m']].itertuples(index=False,name=None)) + '\n'
    text += '''
## 4. 六隔间评测协议

**给定实际环境轨迹的回顾性预测**。三个目标日减 14 个当地日历日，固定当地 17:00 冻结视觉；历史帧从 09-03 00:00（时序起点，排除 09-02 预启动影像）累计到截点（含），RGB 与 RGB-D 必须使用相同且经 QC 的帧交集。截点后实际环境使用到称重日 00:00（不含），因为共同称重时刻未知，避免采样日潜在的测后输入。**不是**提前 14×24 小时，也不是用当天所有照片。目标只用实测株均值；只计隔间×日期，不计终采、重复截点或同日单株。

截点后允许且仅允许下列字段，三种模式共用。`forecast_manifest.csv` 逐字段给出实际非缺测 5 分钟时点数和期望数（密度按 as-of 可查计数，不是独立实测次数）；当前索引可用要求至少一帧、有效目标、截点已知密度及温光有记录，不等于数据连续或全量通过 QC。入模前需固定缺测处理，对三模式使用完全相同的驱动。
'''
    for g, cols in ENV.items():
        text += f'\n- {g}：' + '、'.join(f'`{c}`' for c in cols) + '。'
    if series:
        for field in ENV['inside_measured'][:2]:
            missing = [r.env_expected_5min_ticks - counts[field] for r in forecast.itertuples()
                       if field in (counts := json.loads(r.env_nonmissing_ticks_by_field))]
            if missing:
                text += f'\n- `{field}` 每条回放窗口缺测 {min(missing)}–{max(missing)} 个 5 分钟时点。'
    if 'weather_forecast' in series:
        f = series['weather_forecast']
        text += f'\n\n`weather_forecast.csv` 有 {len(f)} 行；源列：' + '、'.join(f'`{c}`' for c in f if c not in {'utc','local','csv_row'}) + '。没有可验证的签发/版本时间，暂不可用于前瞻评估；也不进入本回顾协议。'
    text += '''

截点后不使用视觉、任何称重/采收日期、生长观测、经济产出或未列字段。密度按实际 CSV 日志 as-of，而非用未来事件回填。留一隔间；只在其余五个隔间或已验证的独立 2023 资料上定参数/阈值，测试隔间称重与观测分布均不得参与拟合或阈值选择。测试样本的环境回放不是参数训练。每行列出其测试折与允许训练的五个隔间。
'''
    text += '\n' + table(['隔间','称重日','截点（当地）','预截点配对','缺时槽','n 株','索引可用','排除原因'],
                         forecast[['compartment','target_date_local','cutoff_local','history_paired_frames','history_missing_nominal_slots','target_n_plants','index_eligible','exclusion_reason']].itertuples(index=False,name=None)) + '\n'
    text += '''
## 5. 2023 预试验的可标定性

**单株 RGB-D。** `pretrial_single_join.csv` 用 `(split, anon_id)` 连接 RGB、Depth、官方 `ground_truth_*.json`，再经 `anon_keys_*.json` 得到原始拍摄日和植株代码。破坏性工作簿用 `(Date, 品种, EC, 光照, Sample n)` 精确联接，不用日期最近邻；缺测不补，日期不跨期修改。同一处理与样本序号跨日可重用，不能视作同株。官方 JSON 的 `height`（cm）、`leaf_area`（cm²）可作为单株观测算子候选标签；`fw_plant` 是全株鲜重，红果数也是另一性状，两者都不是总果实鲜重。绝对图像尺度和土面仍需独立验证，单株受控成像标定不能直接迁移到固定多株俯视。
'''
    if not single.empty:
        linked = single.loc[single.paired_label]
        exact = linked.loc[linked.destructive_candidates.eq(1)]
        matches = exact.destructive_trait_checks.map(json.loads)
        text += f'\n目录级 RGB–Depth–JSON 联表 {len(linked)}/{len(single)}；小样本成功解码 {int(single.decoded.sum())}；匿名键重复行 {int((single.anon_id_repeats>1).sum())}，原始文件名重复行 {int((single.original_filename_repeats>1).sum())}。'
        text += f'与破坏性工作簿唯一键匹配 {len(exact)}，多候选 {int(linked.destructive_candidates.gt(1).sum())}，无候选 {int(linked.destructive_candidates.eq(0).sum())}。'
        for trait in ('height','leaf_area','fw_plant'):
            vals = matches.map(lambda x:x[trait]).value_counts().to_dict()
            text += f' `{trait}` 一致 {vals.get("equal",0)} / 冲突 {vals.get("different",0)} / 缺乏可比值 {vals.get("not_comparable",0)}；'
        disagreements = exact.loc[exact.destructive_trait_checks.str.contains('different')]
        if not disagreements.empty:
            text += '\n\n交叉核对冲突保留双源值，不用于当前标定：\n'
            for r in disagreements.itertuples():
                for field, verdict in json.loads(r.destructive_trait_checks).items():
                    if verdict == 'different':
                        text += f'- `{r.anon_id}` / `{r.original_name}`：`{field}` JSON 为 {getattr(r,field)}，`DestructiveHarvest.xlsx` `All Data` 第 {r.destructive_workbook_rows} 行为 {getattr(r,"workbook_"+field)}。\n'
        text += '\n\n' + table(['原始影像日期','JSON+双图','唯一人工表键','多候选','无候选','不在破坏性人工表日期'],
            ((day,len(g),int(g.destructive_candidates.eq(1).sum()),int(g.destructive_candidates.gt(1).sum()),int(g.destructive_candidates.eq(0).sum()),bool(g.date_not_in_destructive_workbook.all())) for day,g in linked.groupby('image_date'))) + '\n'
        text += '\n09-22、10-09、11-23、11-30 不在本工作簿破坏性日期，后两者也晚于固定冠层归档时段；保留原始日期，不当作同日温室样本，不据此否定官方匿名键标签。\n'
        text += '\n' + table(['已解码单株','拍摄日','像素尺寸','fx px','深度中位原值','约 mm/px @1.5m（示意）'],
            ((r.anon_id,r.image_date,f'{int(r.width)}×{int(r.height)}',r.fx_px,r.depth_raw_median,r.approx_mm_per_px_at_1_5m) for r in linked.loc[linked.decoded].itertuples())) + '\n'
    text += '''
**固定俯视冠层。** 归档 README 仅支持高光处理、画面左 EC6 / 右 EC3；没有给出各盆 ROI、样本标签或调距后映射。日期与高光处理只能做候选分层，**可靠株/ROI 级匹配为 0**，不以同日人工株高或叶面积训练对应像素。相机高度为约数，内参是像素量，也未标定盆顶参考平面。下面的“行数”是候选人工行，不是已配准真值。
'''
    if not can.empty:
        p = can_index.loc[can_index.paired]
        text += f'\n固定相机目录级配对 {len(p)}；未配对或歧义 {len(can_index)-len(p)}；归档实际日期 {p.local_date.min()} 至 {p.local_date.max()}，以实际成员为准，不采用 README 的延续至 12 月表述。成功解码 {int(can.decoded.sum())} 组。'
        text += f' `CropMeasurements.xlsx` 日期×标签重复 {int(crop.duplicated(["Date", "# on label"]).sum())}；破坏性表有代码行日期×代码重复行 {int(manual.loc[manual.plant_code.notna()].duplicated(["day", "plant_code"],keep=False).sum())}（详见 `pretrial_destructive_keys.csv`）。\n'
        text += '\n' + table(['日期','配对帧','高光株高 n','高光破坏性行','高光破坏性叶面积 n','可定位配对'],
            can[['date','image_pairs_on_date','crop_height_n_high_light','destructive_rows_high_light','destructive_leaf_area_n_high_light','reliable_plant_or_roi_links']].itertuples(index=False,name=None)) + '\n'
        if 'fx_px' in can:
            text += '\n' + table(['已解码冠层日','像素尺寸','fx px','深度中位原值','约 mm/px @1.5m'],
                ((r.date,f'{int(r.width)}×{int(r.height)}',r.fx_px,r.depth_raw_median,r.approx_mm_per_px_at_1_5m) for r in can.loc[can.decoded].itertuples())) + '\n'
    text += '''
## 6. 验收与复跑

- **已验证**：发布版本与小文件哈希、真实称重与来源、样本粒度、相机成员配对、DST、截点、日志密度和独立隔间分折；具体下载与解码证据见清单。
- **待验证**：正式赛相机物理深度尺度及盆顶参考平面、像素对齐残差、所有主协议帧的读图与掩码 QC、回放环境缺测策略、调距实际执行时刻、2023 冠层人工样本到画面区域映射。先补校准证据，再验收决定性比较。
- **当前数据不能支持**：H90 的绝对量值、提前 14 日的真实天气预报能力、终采记录作为共同日或逐株纵向样本，以及 RGB-D 已改善果重预测的结论。下一轮通过验收后，在同一机理模型上比较 `开环 / RGB 同化 / RGB-D 同化`，使用相同独立目标逐隔间报告配对 MAE 差异（每隔间最多 3 个目标，不把株或帧作为独立重复）。

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m scripts.phase0 --output audit
.venv/bin/python -m unittest discover -s tests -v
```

首次运行需要网络和可用的 4TU `206` Range；缓存目录 `data/source-cache/` 可复用。需要重验 Range 时，将缓存移走即可。所有主表及补充表由脚本生成；CSV 的空值代表缺失，不是 0。
'''
    text += '\n**运行时来源失败记录**：\n' + ('\n'.join(f'- {e}' for e in failures) if failures else '- 本次无交付件因档案获取、读取或抽样解码失败而缺失。') + '\n'
    (out/'data_audit.md').write_text(text,encoding='utf-8')


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    failures, evidence, source_rows = [], [], []
    statuses = {}
    for name in ('reference_mask_qc.jpg','reference_alignment_qc.jpg'):
        (out/name).unlink(missing_ok=True)

    def attempt(label, operation, fallback=None):
        try:
            return operation()
        except Exception as exc:
            failures.append(f'{label}: {type(exc).__name__}: {exc}')
            return fallback

    for key in ('readme24','series24','series23'):
        data = attempt(f'{key} 下载/MD5',lambda:small(key))
        if data is not None:
            statuses[key] = ('full_md5_verified',hashlib.sha256(data).hexdigest(),None)

    series, book = attempt('2024 时序与称重读取/时区检查；影响真值、密度及清单',load_series,({},None))
    events = density_events(series)
    gt, stat, ledger = (pd.DataFrame(),)*3
    if book is not None:
        gt,stat,ledger = attempt('Harvest.xlsx 解析；影响称重真值及清单',lambda:harvest(book,final_dates(series)),(gt,stat,ledger))

    def camera_index(key, fn):
        with Archive(key) as a:
            statuses[key] = ('range_directory; selected_member_checks_separate',None,len(a.names))
            return fn(a)

    index = pd.DataFrame()
    obs = pd.DataFrame(columns=['compartment','timestamp_local','timestamp_utc','rgb_member','depth_member','plants_m2',
                                'rgb_coverage_fraction','h90_m','mask_valid_depth_fraction','quality_flags'])

    def formal(a):
        readme = a.read('dataset4tu/ReadMe.md')
        evidence.append(a.evidence('dataset4tu/ReadMe.md',readme))
        for comp,(team,cam) in TEAMS.items():
            if not re.search(rf'^\s*{re.escape(comp)}\s+{team}\s+{cam}\s*$',readme.decode(),re.I|re.M):
                raise ValueError(f'{comp}: camera mapping not confirmed in internal README')
        ref_idx = pairs(a.names,{'cam_1'})
        obs_table = attempt('Reference 样帧；影响 compartment_obs.parquet',lambda:reference(a,ref_idx,events,evidence,out,failures),obs)
        return pairs(a.names,{c for _,c in TEAMS.values()}),obs_table

    index,obs = attempt('2024 相机目录/README；影响观测、配对和清单',lambda:camera_index('camera24',formal),(index,obs))
    stats,gaps = gap_audit(index) if not index.empty else ([],pd.DataFrame(columns=['compartment','camera','local_date','hour','issue','n']))
    primary = manifest(index,stat,series,events)
    ref_links = pd.DataFrame()
    if not obs.empty and 'role' in obs:
        ref_targets = primary.loc[primary.compartment.eq('3.06'), ['sample_id','target_date_local',
            'cutoff_local','target_n_plants','target_mean_g_per_plant','target_sd_g_per_plant','target_sheet','target_rows']].copy()
        ref_targets['role'] = 'cutoff_for_' + ref_targets.target_date_local.astype(str)
        ref_links = ref_targets.merge(obs.drop(columns='cutoff_local'), on='role', validate='one_to_one')
    dst = [] if index.empty else [(r.timestamp_local,r.timestamp_utc.isoformat()) for r in index.itertuples()
           if r.camera=='cam_1' and r.hour==4 and str(r.local_date) in {'2024-10-26','2024-10-27'}]

    manual,crop,books23 = attempt('2023 时序/人工工作簿；影响预试验与人工测量联表',pretrial_workbooks,
                                  (pd.DataFrame(columns=['day','plant_code','light','height','leaf_area']),
                                   pd.DataFrame({'Date':pd.Series([],dtype='datetime64[ns]'), 'Light':[], 'Plantheigth':[], '# on label':[]}),{}))
    single = attempt('2023 单株目录与标签；影响预试验单株联表',
                     lambda:camera_index('single23',lambda a:single_plant(a,manual,evidence,failures)),pd.DataFrame())
    can,can_index = attempt('2023 冠层目录与标签；影响预试验冠层日期联表',
                  lambda:camera_index('canopy23',lambda a:canopy(a,manual,crop,evidence,failures)),(pd.DataFrame(),pd.DataFrame()))

    for key,s in SOURCES.items():
        status,sha,members = statuses.get(key,('unavailable',None,None))
        source_rows.append(dict(key=key, doi=f'10.4121/{DOIS[s.year]}.v1', version=1, filename=s.name,
                                url=s.url, bytes=s.size, published_md5=s.md5, sha256=sha, zip_members=members,verification=status))
    for key, names in [('series24',[PREFIX+'Harvest.xlsx',PREFIX+'ReadMe.md',PREFIX+'channel_info.json',
                                     *(PREFIX+'timeseries/'+f'{name}.csv' for name in series)]), ('series23',list(books23))]:
        if key not in statuses:
            continue
        with zipfile.ZipFile(io.BytesIO(small(key))) as z:
            for n in names:
                b = attempt(f'{key}:{n} 成员哈希',lambda:z.read(n))
                if b is not None:
                    evidence.append(dict(archive=key,member=n,bytes=len(b),crc32=f'{z.getinfo(n).CRC:08x}',sha256=hashlib.sha256(b).hexdigest()))
    obs['h90_m'] = obs['h90_m'].astype('Float64')
    obs.to_parquet(out/'compartment_obs.parquet',index=False)
    for name, d in {'harvest_gt':gt,'harvest_summary':stat,'harvest_row_audit':ledger,'forecast_manifest':primary,'reference_cutoff_join':ref_links,
                    'frame_index':index,'frame_gaps':gaps,'density_events':events,'pretrial_single_join':single,
                    'pretrial_canopy_join':can,'pretrial_canopy_frames':can_index,'pretrial_destructive_keys':manual,
                    'source_inventory':pd.DataFrame(source_rows),'verified_members':pd.DataFrame(evidence)}.items():
        d.to_csv(out/f'{name}.csv',index=False)
    report(out,source_rows,gt,stat,ledger,events,obs,primary,stats,gaps,dst,series,single,can,can_index,manual,crop,failures)
    print(f'{len(gt)} individual weights, {len(obs)} Reference samples, {primary.index_eligible.sum()}/{len(primary)} index-eligible targets.')
    return bool(failures)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'audit')
    raise SystemExit(run(parser.parse_args().output))
