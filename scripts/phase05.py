"""Audit the unchanged Phase 0 history windows without reading target weights."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from hashlib import sha256
import io
import json
from pathlib import Path
import threading
import zipfile

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import requests

from .depth_geometry import (candidate_points, canopy_h90, edge_probe, plane_intersection,
                             project, summarize_plane, valid_depth)
from .imaging import MASK_VERSION, rgb_canopy_mask
from .sources import Archive, ROOT, SOURCES, small
from .trial import TEAMS, asof_density

BASELINE = {
    'harvest_gt.csv': '8c7b3ab054912ac2b94724247c410f144fb27c8d674dae4e871a3b02dca313e7',
    'forecast_manifest.csv': 'e0aed0db6f9ce6d6230bb34362e5bf48c5a343f3332d9b16a77445bbdb0c24cd',
    'frame_index.csv': '8fc2dc0ee37bf86f75ee7f7358021afa3c19cd26c8574a88e57b646e133031a3',
    'density_events.csv': '49967ac31877611c9726a5195538d631ca379ee6147d3a4e2c70aa4082d1be50',
    'compartment_obs.parquet': 'b1facd0af35c02585beebee993be8ed3f4927f041f17620be945dd0b6afbca2b',
}
MANIFEST_COLUMNS = ['sample_id', 'compartment', 'camera', 'target_date_local', 'cutoff_local', 'cutoff_utc',
                    'history_start_local', 'history_paired_frames', 'holdout_fold', 'test_compartment',
                    'train_compartments', 'environment_protocol', 'index_eligible']
DATA_ERRORS = (OSError, ValueError, zipfile.BadZipFile, requests.RequestException)
DEPTH_SCALE = .001
MIN_DEPTH_FRACTION = .5  # Phase 0 diagnostic QC; not a calibrated acceptance rule.
NOMINAL_POT_DIAMETER_M = .13  # Maree et al., Sensors 25 (2025) 4321, section 2.2.
ARCHIVE_MEMBER = 'dataset4tu/ReadMe.md'
BASE_DIR = ROOT / 'audit'


def in_window(frames, target):
    return (frames.camera.eq(target.camera) & frames.timestamp_utc.ge(pd.Timestamp(target.history_start_local))
            & frames.timestamp_utc.le(target.cutoff_utc))


def load_scope(base):
    for name, expected in BASELINE.items():
        if sha256((base / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'{name}: not byte-identical to Phase 0 faf2852')
    manifest = pd.read_csv(base / 'forecast_manifest.csv', dtype={'compartment': str, 'test_compartment': str},
                           usecols=MANIFEST_COLUMNS)
    manifest['cutoff_utc'] = pd.to_datetime(manifest.cutoff_utc, utc=True)
    if len(manifest) != 18 or manifest.sample_id.duplicated().any():
        raise ValueError('Unexpected target set')
    frames = pd.read_csv(base / 'frame_index.csv', keep_default_na=False)
    frames['timestamp_utc'] = pd.to_datetime(frames.timestamp_utc, utc=True)
    union = pd.Series(False, index=frames.index)
    for target in manifest.itertuples():
        window = in_window(frames, target)
        if (window & frames.paired).sum() != target.history_paired_frames:
            raise ValueError(f'Phase 0 index mismatch: {target.sample_id}')
        union |= window
    frames = frames[union].copy()
    if frames.duplicated(['camera', 'timestamp_utc']).any():
        raise ValueError('Duplicate camera/time in history')
    frames['frame_id'] = frames.rgb_members.str.extract(r'(?:^|/)([^/]+)_rgb\.png$')[0].fillna(
        frames.depth_members.str.extract(r'(?:^|/)([^/]+)_depth\.png$')[0])
    frames['compartment'] = frames.camera.map({cam: comp for comp, (_, cam) in TEAMS.items()})
    if frames.frame_id.isna().any() or frames.compartment.isna().any():
        raise ValueError('Unidentified frame or compartment')
    events = pd.read_csv(base / 'density_events.csv', dtype={'compartment': str}, keep_default_na=False)
    events['utc'] = pd.to_datetime(events.utc, utc=True)
    events = events[events.valid].sort_values('utc')
    events['change'] = events.groupby('compartment').plants_m2.diff().fillna(0).ne(0)
    intervals = [('w1_to_first_cutoff', '2024-09-23'), ('w2_to_second_cutoff', '2024-10-08'),
                 ('w3_to_last_cutoff', '2024-10-22')]
    for field, day in reversed(intervals):
        frames.loc[frames.local_date.le(day), 'history_stage'] = field
    for frame in frames.itertuples():
        event = asof_density(events, frame.compartment, frame.timestamp_utc)
        if event is not None:
            frames.loc[frame.Index, 'density_plants_m2'] = event.plants_m2
            frames.loc[frame.Index, 'density_log_timestamp_local'] = event.log_timestamp_local
        local_events = events[events.compartment.eq(frame.compartment)]
        frames.loc[frame.Index, 'logged_density_change_day'] = bool(
            (local_events.change & local_events.log_date_local.eq(frame.local_date)).any())
        frames.loc[frame.Index, 'documented_move_day'] = bool(local_events.actual_move_date_local.eq(frame.local_date).any())
    return frames.sort_values(['compartment', 'timestamp_utc']), manifest


def sample_plan(frames, ref_ids):
    reference = frames[frames.frame_id.isin(ref_ids)]
    if set(reference.frame_id) != set(ref_ids):
        raise ValueError('Missing Phase 0 Reference samples')
    others = []
    for _, group in frames[frames.camera.ne('cam_1') & frames.paired].groupby('camera'):
        # Early and last daylight views + the frames on both sides of one logged change.
        daylight = group[group.hour.eq(16)]
        selected_event = daylight[daylight.logged_density_change_day]
        if not selected_event.empty:
            event_date = selected_event.iloc[0].local_date
            others.extend(daylight[daylight.local_date.lt(event_date)].tail(1).frame_id)
            others.append(selected_event.iloc[0].frame_id)
        others.extend(daylight[daylight.local_date.eq('2024-09-04')].frame_id)
        others.append(daylight.iloc[-1].frame_id)
    return list(reference.frame_id), list(dict.fromkeys(others))


def read_configuration(archive):
    readme = archive.read(ARCHIVE_MEMBER)
    root_readme = small('readme24').decode('utf-8')
    text = readme.decode('utf-8')
    configs, evidence = {}, {}
    for comp, (team, cam) in TEAMS.items():
        if not any(line.lower().split() == [comp, team, cam] for line in text.splitlines()):
            raise ValueError(f'{cam}: archive README mapping mismatch')
        member = f'dataset4tu/configs/{cam}.json'
        data = archive.read(member)
        cfg = json.loads(data)
        k = cfg['color_int']
        if cfg['name'] != 'Oak-D S2 POE' or (k['width'], k['height']) != (3840, 2160):
            raise ValueError(f'{cam}: unexpected camera or RGB grid')
        if not (np.isfinite([k['fx'], k['fy'], k['ppx'], k['ppy']]).all()
                and min(k['fx'], k['fy']) > 0 and 0 <= k['ppx'] < 3840 and 0 <= k['ppy'] < 2160):
            raise ValueError(f'{cam}: invalid color intrinsics')
        if cfg.get('DepthScale', DEPTH_SCALE) != DEPTH_SCALE:
            raise ValueError(f'{cam}: camera-specific depth scale supersedes this audit')
        configs[cam] = cfg
        evidence[cam] = dict(**archive.evidence(member, data), intrinsics=k, camera_model=cfg['name'],
                             explicit_scale=cfg.get('DepthScale'), auto_focus=cfg.get('auto_focus'),
                             lens_position=cfg.get('lens_position'))
    if any('aligned to the color images' not in readme for readme in (text, root_readme)):
        raise ValueError('Archive alignment declaration missing')
    return configs, evidence, archive.evidence(ARCHIVE_MEMBER, readme)


def load_image(archive, name, mode, size, row, label, cache_only):
    try:
        data = archive.read(name, cache_only=cache_only)
        row[f'{label}_member_sha256'] = sha256(data).hexdigest()
        with Image.open(io.BytesIO(data)) as img:
            if img.format != 'PNG' or img.mode != mode or img.size != size:
                raise ValueError(f'unexpected {img.format}/{img.mode}/{img.size}')
            # Cast is deliberately avoided so UINT8/truncated images cannot pass as depth.
            array = np.asarray(img)
        if label == 'depth' and array.dtype != np.uint16:
            raise ValueError(f'depth dtype {array.dtype} is not uint16')
        row[f'{label}_decoded'] = True
        return array
    except DATA_ERRORS as exc:
        row[f'{label}_error'] = f'{type(exc).__name__}: {exc}'
        return None


def sample_geometry(raw, mask, k, annotation, row):
    # Explicitly rejected longitudinally: these fits are not accepted H90 reference planes.
    row['reference_plane_status'] = annotation['status']
    row['annotation_note'] = annotation['note']
    for surface in ('pot_rim', 'bench'):
        if surface not in annotation:
            continue
        boxes = [[3*c for c in box] for box in annotation[surface]]
        fit = summarize_plane(list(candidate_points(raw, mask, boxes, k, DEPTH_SCALE)))
        if 'normal' in fit:
            fit['z_at_optical_axis_m'] = -fit['d_m'] / fit['normal'][2]
        row[f'{surface}_fit'] = json.dumps(dict(**fit, native_boxes=boxes), sort_keys=True)
    if row.get('pot_rim_fit') and row.get('bench_fit'):
        rim, bench = (json.loads(row[f'{s}_fit']) for s in ('pot_rim', 'bench'))
        if 'normal' in rim and 'normal' in bench:
            # Compare fitted local surfaces at the same RGB ray in their vicinity (not a verified offset).
            x0,y0,x1,y1 = rim['native_boxes'][-1]
            u,v = (x0+x1)/2, (y0+y1)/2
            point = plane_intersection(u, v, k, np.array(rim['normal']), rim['d_m'])
            row['bench_to_rim_apparent_mm'] = float((point @ bench['normal'] + bench['d_m'])*1000)
            row['rim_bench_normal_angle_deg'] = float(np.degrees(np.arccos(np.clip(
                np.dot(rim['normal'], bench['normal']), -1, 1))))
    if 'diameter_chords' in annotation and row.get('pot_rim_fit'):
        rim = json.loads(row['pot_rim_fit'])
        if 'normal' in rim:
            diameters = []
            for chord in np.asarray(annotation['diameter_chords']) * 3:
                n, d = np.array(rim['normal']), rim['d_m']
                pts = plane_intersection(*chord.T, k, n, d)
                length = float(np.linalg.norm(pts[0] - pts[1]))
                # One-pixel rays are brittle at the boundary: report a +/-3 native px endpoint check.
                lengths = [np.linalg.norm(np.diff(plane_intersection(chord[:, 0], chord[:, 1]+[s,-s], k,n,d), axis=0))
                           for s in (-3, 3)]
                diameters.append(dict(native_chord=chord.tolist(), predicted_mm=length*1000,
                                      nominal_mm=NOMINAL_POT_DIAMETER_M*1000,
                                      difference_mm=(length-NOMINAL_POT_DIAMETER_M)*1000,
                                      endpoint_sensitivity_mm=sorted(v*1000 for v in lengths)))
            row['diameter_checks'] = json.dumps(diameters, sort_keys=True)
    if 'outward_y_edges' in annotation:
        row['registration_edge_checks'] = json.dumps([
            dict(native_rgb_edge=[3*x, 3*y], **edge_probe(raw, 3*x, 3*y))
            for x, y in annotation['outward_y_edges']], sort_keys=True)


def diagnostic(path, rgb, raw, mask, row):
    w, h = 768, 432
    bar = 24
    panel = Image.new('RGB', (w*3, h+2*bar), (20, 20, 20))
    d = ImageDraw.Draw(panel)
    d.text((9, 5), f'{row["frame_id"]} | {row["reference_plane_status"]} | H90 unavailable', fill='white')
    if rgb is not None:
        base = Image.fromarray(rgb)
        panel.paste(base.resize((w, h)), (0, bar))
        tinted = base.resize((w, h)).convert('RGB')
        m = Image.fromarray(mask).resize((w, h), Image.Resampling.NEAREST)
        tinted.paste(Image.blend(tinted, Image.new('RGB', (w, h), '#16ff49'), .45), mask=m)
        panel.paste(tinted, (w, bar))
    if raw is not None:
        grid = raw[::5,::5]
        good = valid_depth(grid)
        if good.any():
            lo, hi = np.percentile(grid[good], [2, 98])
            z = np.where(good, 40+215*np.clip((grid.astype(float)-lo)/max(hi-lo, 1), 0, 1), 0)
            panel.paste(Image.fromarray(z.astype(np.uint8)).convert('RGB'), (w*2, bar))
    for surface, color in [('pot_rim', '#ff4858'), ('bench', '#28e8ff')]:
        if row.get(f'{surface}_fit'):
            for x0, y0, x1, y1 in json.loads(row[f'{surface}_fit'])['native_boxes']:
                for panel_id in (0, 1, 2):
                    d.rectangle((panel_id*w+x0/5, bar+y0/5, panel_id*w+x1/5, bar+y1/5), outline=color, width=2)
    if row.get('diameter_checks'):
        for trial in json.loads(row['diameter_checks']):
            d.line([(x/5, bar+y/5) for x, y in trial['native_chord']], fill='#ffda38', width=2)
    for i, title in enumerate(['RGB: red rim / cyan bench / yellow chords', 'Same RGB mask used for both arms',
                               'Axial depth: per-frame 2-98% stretch; invalid black']):
        d.text((i*w+9, h+bar+5), title, fill='white')
    panel.save(path, quality=85)


def audit_frame(frame, archive, config, config_evidence, annotation=None, picture=None, *, cache_only=False):
    k = config['color_int']
    size = (k['width'], k['height'])
    row = dict(attempted=True, rgb_decoded=False, depth_decoded=False, rgb_screen_ok=False,
               common_screen_ok=False, h90_qualified=False, h90_m=None,
               depth_scale_m_per_raw_unit=DEPTH_SCALE, config_member=config_evidence['member'],
               config_sha256=config_evidence['sha256'], scale_status='encoding_verified_field_accuracy_not_accepted',
               reference_plane_status='not_validated_no_per_frame_reference', registration_status='not_checked',
               rgb_mask_status='not_available', rgb_coverage_fraction=None, mask_valid_depth_fraction=None,
               mask_version=MASK_VERSION, rgb_roi='0,0,3840,2160')
    rgb = load_image(archive, frame.rgb_members, 'RGB', size, row, 'rgb', cache_only) if frame.rgb_n == 1 else None
    raw = load_image(archive, frame.depth_members, 'I;16', size, row, 'depth', cache_only) if frame.depth_n == 1 else None
    mask = None
    failure = []
    if rgb is not None:
        mask = rgb_canopy_mask(rgb)
        n = int(mask.sum())
        row['mask_pixels'] = n
        row['rgb_coverage_fraction'] = n / mask.size
        dark = float((rgb[::8,::8].max(axis=2) < 16).mean())
        row['rgb_near_black_fraction_sampled'] = dark
        row['rgb_mask_status'] = ('empty' if not n else 'near_black' if dark >= .95 else
                                  'nonempty_screen_only_no_semantic_acceptance')
        row['rgb_screen_ok'] = bool(n and dark < .95)
        if not row['rgb_screen_ok']:
            failure.append(row['rgb_mask_status'])
    else:
        failure.append('rgb_unavailable')
    if raw is None:
        failure.append('depth_unavailable')
    else:
        valid = valid_depth(raw)
        row['depth_zeros'] = int((raw == 0).sum())
        row['depth_all_ones'] = int((raw == 65535).sum())
        row['depth_valid_fraction'] = float(valid.mean())
        if rgb is not None:
            row['registration_status'] = 'archive_declared_and_grid_equal_only'
            if n:
                pixel_values = raw[mask & valid]
                fraction = len(pixel_values)/n
                row['mask_valid_depth_fraction'] = fraction
                if len(pixel_values):
                    row['masked_z_median_raw'] = float(np.median(pixel_values))
                if fraction < MIN_DEPTH_FRACTION:
                    failure.append('insufficient_valid_depth_in_mask')
                row['common_screen_ok'] = bool(frame.paired and row['rgb_screen_ok'] and fraction >= MIN_DEPTH_FRACTION)
        if mask is not None and annotation is not None:
            sample_geometry(raw, mask, k, annotation, row)
            if row.get('registration_edge_checks'):
                row['registration_status'] = 'grid_equal_local_natural_edges_probed_no_calibration_grid'
    # No field-certified scale and no portable pot-top plane were established.
    if raw is not None and mask is not None:
        row['h90_m'] = canopy_h90(raw, mask, k, DEPTH_SCALE, None, scale_passed=False, plane_passed=False)
    failure += ['field_scale_not_accepted', row['reference_plane_status']]
    row['failure_reasons'] = ';'.join(failure)
    if picture is not None:
        diagnostic(picture, rgb, raw, mask, dict(frame_id=frame.frame_id, **row))
        row['diagnostic_file'] = str(Path(picture.parent.name) / picture.name)
    return row


def coverage_table(quality, manifest):
    rows = []
    for target in manifest.itertuples(index=False):
        group = quality[in_window(quality, target)]
        passed = int(group.h90_qualified.sum())
        rows.append(dict(**target._asdict(), directory_frames=len(group), directory_pairs=int(group.paired.sum()),
                         attempted_frames=int(group.attempted.sum()), unattempted_frames=int((~group.attempted).sum()),
                         rgb_decoded_frames=int(group.rgb_decoded.sum()), depth_decoded_frames=int(group.depth_decoded.sum()),
                         decoded_pairs=int((group.paired & group.rgb_decoded & group.depth_decoded).sum()),
                         available_rgb_frames=int(group.rgb_screen_ok.sum()), common_rgbd_frames=int(group.common_screen_ok.sum()),
                         qualified_h90_frames=passed, paired_evaluation_eligible=bool(target.index_eligible and passed > 0),
                         eligibility_reason='' if passed else 'no_qualified_h90_before_cutoff'))
    return pd.DataFrame(rows)


def execute(args):
    frames, manifest = load_scope(args.baseline)
    annotations = json.loads(args.annotations.read_text())['frames']
    pilot_obs = pd.read_parquet(args.baseline / 'compartment_obs.parquet')
    ref_ids = set(frames.loc[frames.rgb_members.isin(pilot_obs.rgb_member), 'frame_id'])
    if set(annotations) != ref_ids:
        raise ValueError('Annotations and Phase 0 observation sample disagree')
    ref_ids, others = sample_plan(frames, annotations)
    by_id = {row.frame_id: row for row in frames.itertuples(index=False)}
    args.output.mkdir(parents=True, exist_ok=True)
    images = args.output / 'phase05_diagnostics'
    images.mkdir(exist_ok=True)
    results = {}
    with Archive('camera24') as archive:
        configs, cal, readme = read_configuration(archive)
        names = frames.loc[frames.paired, ['rgb_members', 'depth_members']].to_numpy().ravel()
        if len(set(names)) != len(names):
            raise ValueError('Archive member reused across history rows')
        archive_bytes = sum(archive.z.getinfo(name).compress_size for name in names)
        print(f'History: {len(frames)} indexed frames, {archive_bytes:,} compressed pair member bytes', flush=True)
        for ids in (ref_ids, [] if args.scope == 'reference' else others):
            for ident in ids:
                row = by_id[ident]
                results[ident] = audit_frame(row, archive, configs[row.camera], cal[row.camera],
                                             annotations.get(ident), images / (ident + '.jpg'), cache_only=args.cache_only)
            if ids:
                print(f'Pilot: {len(results)} frames attempted', flush=True)

    if args.scope == 'full':
        # Interleave compartments; no duplicate decodes even across overlapping targets.
        rest = frames[~frames.frame_id.isin(results)].sort_values(['timestamp_utc', 'camera'])
        if args.extra_per_camera is not None:
            rest = rest.groupby('camera').head(args.extra_per_camera)
        local = threading.local()
        with ExitStack() as archives:
            lock = threading.Lock()

            def worker(row):
                if not hasattr(local, 'archive'):
                    with lock:
                        local.archive = archives.enter_context(Archive('camera24'))
                return row.frame_id, audit_frame(row, local.archive, configs[row.camera], cal[row.camera],
                                                 cache_only=args.cache_only)

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                tasks = [pool.submit(worker, row) for row in rest.itertuples(index=False)]
                for count, task in enumerate(as_completed(tasks), 1):
                    ident, result = task.result()
                    results[ident] = result
                    if count % 50 == 0 or count == len(rest):
                        print(f'History: {count}/{len(rest)} additional frames attempted', flush=True)

    pending = dict(attempted=False, rgb_decoded=False, depth_decoded=False, rgb_screen_ok=False,
                   common_screen_ok=False, h90_qualified=False, h90_m=None, rgb_coverage_fraction=None,
                   mask_valid_depth_fraction=None, scale_status='not_checked', reference_plane_status='not_checked',
                   rgb_mask_status='not_checked', failure_reasons='not_attempted_scope_limit')
    extra = pd.DataFrame([results.get(ident, pending) for ident in frames.frame_id])
    quality = pd.concat([frames.reset_index(drop=True), extra], axis=1)
    for col in ('rgb_coverage_fraction', 'mask_valid_depth_fraction', 'h90_m'):
        quality[col] = quality[col].astype(float)
    coverage = coverage_table(quality, manifest)
    quality.to_parquet(args.output / 'frame_quality.parquet', index=False)
    coverage.to_csv(args.output / 'forecast_coverage.csv', index=False)
    strata = []
    for level in ('compartment', 'history_stage', 'logged_density_change_day', 'documented_move_day', 'hour'):
        for name, group in quality.groupby(level):
            strata.append(dict(stratum=level, value=name, directory_pairs=int(group.paired.sum()),
                               attempted=int(group.attempted.sum()), rgb_available=int(group.rgb_screen_ok.sum()),
                               common_rgbd=int(group.common_screen_ok.sum()), h90=int(group.h90_qualified.sum())))
    pd.DataFrame(strata).to_csv(args.output / 'phase05_quality_strata.csv', index=False)
    evidence = dict(archive_version='10.4121/fa102772-32db-4b30-bace-12f2016722ce.v1',
                    archive_name=SOURCES['camera24'].name, archive_published_md5=SOURCES['camera24'].md5,
                    archive_whole_md5_verified=False, compressed_history_member_bytes=archive_bytes,
                    archive_readme=readme, cameras=cal, baseline_faf2852_sha256=BASELINE,
                    annotations_sha256=sha256(args.annotations.read_bytes()).hexdigest())
    (args.output / 'phase05_source_evidence.json').write_text(json.dumps(evidence, indent=2, sort_keys=True)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=BASE_DIR)
    parser.add_argument('--annotations', type=Path, default=BASE_DIR / 'phase05_annotations.json')
    parser.add_argument('--output', type=Path, default=BASE_DIR)
    parser.add_argument('--scope', choices=['reference', 'pilot', 'full'], default='full')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--cache-only', action='store_true', help='Do not request uncached image members')
    parser.add_argument('--extra-per-camera', type=int, help='Optional historical audit limit after pilot')
    args = parser.parse_args()
    if args.workers < 1 or args.extra_per_camera is not None and args.extra_per_camera < 0:
        parser.error('workers must be positive and extra-per-camera cannot be negative')
    execute(args)


if __name__ == '__main__':
    main()
