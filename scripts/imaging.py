"""RGB-only audit mask and measurement-link checks for the two trials."""

import io
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from datetime import date, datetime

import numpy as np
import openpyxl
import pandas as pd
from PIL import Image

from .sources import Archive, small
from .trial import DAYS, asof_density, cutoff, pairs

REF_CONFIG = 'dataset4tu/configs/cam_1.json'
DEMO_COMMIT = '86c0b60cb55ca197b86c74394492c5b6612aff55'
VARIETY = {'cherry': 'ch', 'heart': 'he', 'plum': 'pl', 'nc': 'nc'}
TRAITS = ('height', 'leaf_area', 'fw_plant', 'number_of_red_fruits')


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def decode(archive, rgb_name, depth_name, config, evidence, *, mask=False):
    rgb_bytes, dep_bytes = (archive.read(n) for n in (rgb_name, depth_name))
    evidence.extend(archive.evidence(n, b) for n,b in ((rgb_name,rgb_bytes),(depth_name,dep_bytes)))
    with Image.open(io.BytesIO(rgb_bytes)) as img:
        if img.format != 'PNG' or img.mode != 'RGB':
            raise ValueError(f'unexpected RGB format/mode: {img.format}/{img.mode}')
        rgb = np.asarray(img)
    with Image.open(io.BytesIO(dep_bytes)) as img:
        if img.format != 'PNG':
            raise ValueError(f'unexpected depth format: {img.format}')
        raw = np.asarray(img)
    if raw.dtype != np.uint16 or raw.shape != rgb.shape[:2]:
        raise ValueError(f'depth mismatch: {raw.shape}/{raw.dtype}; RGB: {rgb.shape}')
    intr = config['color_int']
    h, w = raw.shape
    if (w,h) != (intr['width'],intr['height']):
        raise ValueError(f'{rgb_name}: image grid mismatches color_int')
    valid = (raw > 0) & (raw < 65535)
    raw_z = raw[valid]
    row = dict(width=w, height=h, fx_px=intr['fx'], fy_px=intr['fy'], cx_px=intr['ppx'], cy_px=intr['ppy'],
               depth_scale_m_per_raw_unit=None, scale_status='not_verified_no_explicit_scale_or_metric_target',
               assumed_demo_scale_m_per_unit=.001,
               approx_mm_per_px_at_1_5m=1500 / intr['fx'],
               depth_raw_min=int(raw_z.min()) if len(raw_z) else None,
               depth_raw_median=float(np.median(raw_z)) if len(raw_z) else None,
               depth_raw_max=int(raw_z.max()) if len(raw_z) else None,
               depth_zero_fraction=float((raw == 0).mean()), depth_valid_fraction=float(valid.mean()),
               alignment_status='archive_declared_aligned_and_pixel_grid_verified_no_target_residual_test')
    selection = None
    if mask:
        # Fixed RGB-only screening proxy. No target, depth or held-out statistics fit this rule.
        r,g,b = (rgb[:,:,i].astype(np.int16) for i in range(3))
        selection = (g - r >= 6) & (g - b >= -12) & (g >= 25)
        n = int(selection.sum())
        row.update(roi=f'0,0,{w},{h}', roi_convention='x0,y0,x1,y1_exclusive',
                   mask_version='rgb_green_proxy_v1:G-R>=6;G-B>=-12;G>=25', canopy_pixel_n=n,
                   rgb_coverage_fraction=n/(w*h), mask_valid_depth_fraction=int((selection & valid).sum())/n if n else None)
    return row, rgb, selection, raw


def reference(archive, index, events, evidence, outdir, failures):
    config_raw = archive.read(REF_CONFIG)
    config = json.loads(config_raw)
    version = archive.evidence(REF_CONFIG, config_raw)
    evidence.append(version)
    ref = index.loc[index.paired & index.camera.eq('cam_1')].sort_values('timestamp_utc')
    chosen = []
    for day in DAYS:
        stop = cutoff(day)
        chosen.append((f'cutoff_for_{day}', ref.loc[ref.timestamp_utc <= stop].iloc[-1], stop.isoformat()))
    # Last daytime frame on either side of the 30 -> 20 log event, not asserted as physical move time.
    for d, role in [(date(2024,10,6), 'before_density_log'), (date(2024,10,7), 'after_density_log'),
                    (date(2024,9,4), 'early_plane_inspection_only')]:
        selection = ref.loc[ref.local_date.eq(d) & ref.hour.eq(16)]
        if not selection.empty:
            chosen.append((role, selection.iloc[0], None))
    rows = []
    previews = []
    for role, frame, stop in chosen:
        density = asof_density(events,'3.06',frame.timestamp_utc)
        row = dict(compartment='3.06', camera='cam_1', role=role, cutoff_local=stop,
                   timestamp_local=frame.timestamp_local, timestamp_utc=frame.timestamp_utc,
                   rgb_member=frame.rgb_members, depth_member=frame.depth_members, source_archive='camera24',
                   plants_m2=density.plants_m2 if density is not None else None,
                   density_log_timestamp_local=density.log_timestamp_local if density is not None else None,
                   actual_move_timestamp=None, actual_move_date_local=None, actual_move_status='not_verified_no_physical_move_log',
                   calibration_member=REF_CONFIG, calibration_sha256=version['sha256'],
                   reference_plane_status='not_verified_no_bench_or_pot_soil_plane', h90_m=None,
                   h90_missing_reason='metric_depth_scale_and_reference_plane_not_verified',
                   rgb_coverage_fraction=None, mask_valid_depth_fraction=None, roi=None, mask_version=None,
                   quality_flags='pilot_mask_not_calibrated;metric_scale_unverified;reference_plane_unverified')
        try:
            metrics,rgb,selected,raw = decode(archive,frame.rgb_members,frame.depth_members,config,evidence,mask=True)
            row.update(metrics)
            if not metrics['canopy_pixel_n']:
                row['quality_flags'] += ';empty_mask'
            elif metrics['mask_valid_depth_fraction'] < .5:
                row['quality_flags'] += ';less_than_half_mask_with_valid_depth'
            image = Image.fromarray(rgb).resize((480,270))
            suppressed = np.where(selected[:,:,None], rgb, rgb//4).astype(np.uint8)
            overlay = Image.fromarray(suppressed).resize((480,270))
            previews.append((image, overlay))
            if role == 'early_plane_inspection_only':
                good = (raw > 0) & (raw < 65535)
                lo,hi = np.percentile(raw[good], [2,98])
                shade = np.where(good, 30 + 225*np.clip((raw-lo)/max(1,hi-lo),0,1), 0).astype(np.uint8)
                sheet = Image.new('RGB',(1280,360))
                sheet.paste(Image.fromarray(rgb).resize((640,360)),(0,0))
                sheet.paste(Image.fromarray(shade).resize((640,360)),(640,0))
                sheet.save(outdir/'reference_alignment_qc.jpg',quality=88)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            failures.append(f'Reference {role}: {type(exc).__name__}: {exc}')
            row['quality_flags'] += ';decode_failed'
        rows.append(row)
    if previews:
        preview = Image.new('RGB',(960,270*len(previews)))
        for i,(im,overlay) in enumerate(previews):
            preview.paste(im,(0,i*270))
            preview.paste(overlay,(480,i*270))
        preview.save(outdir/'reference_mask_qc.jpg',quality=86)
    return pd.DataFrame(rows)


def pretrial_workbooks():
    with zipfile.ZipFile(io.BytesIO(small('series23'))) as z:
        files = {n:z.read(n) for n in z.namelist() if n.endswith('.xlsx')}
    d = openpyxl.load_workbook(io.BytesIO(files['DestructiveHarvest.xlsx']),read_only=True,data_only=True)
    all_rows = list(d['All Data'].values)
    layout = {str(x).strip():i for i,x in enumerate(all_rows[4]) if x is not None}
    records = []
    for i,r in enumerate(all_rows[5:],6):
        def col(k): return r[layout[k]]
        day, variety, light, ec, n = (col(k) for k in ('Date','Variety','Light','EC','Sample n'))
        if not isinstance(day,datetime):
            continue
        v = VARIETY.get(str(variety).lower())
        if v and isinstance(n,int) and re.fullmatch(r'EC[36]',str(ec)) and str(light).lower()[0] in 'hmln':
            key = f'{v}{str(ec)[-1]}{str(light).lower()[0]}{n}'
        else:
            key = None
        records.append(dict(day=day.date(), plant_code=key, source_row=i, variety=variety, ec=ec, light=light,
                            height=col('Plant height'), leaf_area=col('Leaf Area measured (cm²/plant)'),
                            fw_plant=col('FW Whole plants measured (g/plant)'), fruit_fw=col('FW fruits clusters(g/plant)')))
    d.close()
    crop = pd.read_excel(io.BytesIO(files['CropMeasurements.xlsx']),sheet_name='All data')
    return pd.DataFrame(records), crop, files


def single_plant(archive, destructive, evidence, failures):
    def member(name):
        data = archive.read(name)
        evidence.append(archive.evidence(name,data))
        return data
    member('single_plant_cam/ReadMe.md')
    config_name = 'single_plant_cam/oak-d-s2-poe_conf.json'
    config = json.loads(member(config_name))
    key_counts, name_counts = Counter(), Counter()
    entries = []
    groups = defaultdict(list)
    for r in destructive.itertuples():
        if r.plant_code:
            groups[r.day,r.plant_code].append(r)
    for split in ('train','validation1','validation2','validation3','test'):
        base = f'single_plant_cam/{split}'
        gt_path = f'{base}/ground_truth_{split}.json'
        anon_path = f'{base}/anon_keys_{split}.json'
        gt = json.loads(member(gt_path))
        anon = json.loads(member(anon_path))
        for key in sorted(gt.keys() | anon.keys()):
            key_counts[key] += 1
            raw_name = anon.get(key,'')
            name_counts[raw_name] += 1
            m = re.fullmatch(r'camir_27_(\d{4}(?:_\d{2}){5})_([a-z]{2}[36][hmln]\d+)_rgb',raw_name)
            day = datetime.strptime(m[1],'%Y_%m_%d_%H_%M_%S').date() if m else None
            code = m[2] if m else None
            rgb, dep = f'{base}/rgb/{key}.png', f'{base}/depth/{key}_depth.png'
            n_rgb, n_dep = archive.names.count(rgb), archive.names.count(dep)
            matching = groups[day,code]
            trait = gt.get(key,{})
            cmp = {c: ('equal' if math.isclose(getattr(matching[0],c),trait[c],abs_tol=.011,rel_tol=0) else 'different')
                   if len(matching)==1 and finite(getattr(matching[0],c)) and finite(trait.get(c)) else 'not_comparable'
                   for c in ('height','leaf_area','fw_plant')}
            entries.append(dict(split=split, anon_id=key, ground_truth_member=gt_path, anon_key_member=anon_path,
                                original_name=raw_name, image_date=day, plant_code=code, rgb_member=rgb, depth_member=dep,
                                rgb_n=n_rgb, depth_n=n_dep, in_ground_truth=key in gt, paired_label=n_rgb==n_dep==1 and key in gt,
                                **{k:trait.get(k) if finite(trait.get(k)) else None for k in TRAITS},
                                **{f'workbook_{k}':getattr(matching[0],k) if len(matching)==1 else None
                                   for k in ('height','leaf_area','fw_plant','fruit_fw')},
                                destructive_candidates=len(matching), destructive_workbook_rows=';'.join(str(r.source_row) for r in matching),
                                destructive_trait_checks=json.dumps(cmp,sort_keys=True),
                                date_not_in_destructive_workbook=day not in set(destructive.day),
                                decoded=False))
    sampled = set()
    for row in entries:
        day = row['image_date']
        if not row['paired_label'] or day is None or day in sampled:
            continue
        # Three exact links at distinct workbook dates and one out-of-workbook date.
        if day not in {date(2023,9,18),date(2023,9,29),date(2023,11,9),date(2023,11,23)}:
            continue
        if day != date(2023,11,23) and row['destructive_candidates']!=1:
            continue
        sampled.add(day)
        try:
            metrics,_,_,_ = decode(archive,row['rgb_member'],row['depth_member'],config,evidence)
            row.update(metrics,decoded=True, calibration_member=config_name)
        except (OSError,ValueError,zipfile.BadZipFile) as exc:
            failures.append(f'2023 single {row["anon_id"]}: {type(exc).__name__}: {exc}')
    for row in entries:
        row['anon_id_repeats'] = key_counts[row['anon_id']]
        row['original_filename_repeats'] = name_counts[row['original_name']]
    return pd.DataFrame(entries)


def canopy(archive, destructive, crop, evidence, failures):
    def member(name):
        data = archive.read(name)
        evidence.append(archive.evidence(name,data))
        return data
    member('canopy_cam/ReadMe.md')
    cfg = 'canopy_cam/canopy_camera_config.json'
    config = json.loads(member(cfg))
    images = pairs(archive.names, {'cam_19'})
    rows = []
    days = sorted(set(crop.Date.dt.date) | set(destructive.day) |
                  {date(2023,9,18),date(2023,9,29),date(2023,11,9)})
    for d in days:
        cam = images.loc[images.local_date.eq(d) & images.paired]
        dest = destructive.loc[destructive.day.eq(d) & destructive.light.astype(str).str.lower().eq('high light')]
        manual = crop.loc[crop.Date.dt.date.eq(d) & crop.Light.astype(str).str.lower().eq('high light')]
        row = dict(date=d, image_pairs_on_date=len(cam), crop_rows_high_light=len(manual),
                   crop_height_n_high_light=int(manual.Plantheigth.notna().sum()),
                   destructive_rows_high_light=len(dest), destructive_height_n_high_light=int(dest.height.notna().sum()),
                   destructive_leaf_area_n_high_light=int(dest.leaf_area.notna().sum()),
                   match_key='date + documented high-light stratum only', reliable_plant_or_roi_links=0,
                   decoded=False, calibration_member=cfg)
        if d in {date(2023,9,18),date(2023,9,29),date(2023,11,9)} and not cam.empty:
            frame = cam.loc[(cam.hour-12).abs().idxmin()]
            row.update(timestamp_local=frame.timestamp_local,rgb_member=frame.rgb_members,depth_member=frame.depth_members)
            try:
                metrics,_,_,_ = decode(archive,frame.rgb_members,frame.depth_members,config,evidence)
                row.update(metrics,decoded=True)
            except (OSError,ValueError,zipfile.BadZipFile) as exc:
                failures.append(f'2023 canopy {d}: {type(exc).__name__}: {exc}')
        rows.append(row)
    return pd.DataFrame(rows), images
