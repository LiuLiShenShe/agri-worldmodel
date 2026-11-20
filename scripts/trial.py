"""2024 trial: outcome provenance, cutoff-safe image indexing and covariates."""

import io
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import openpyxl
import pandas as pd

from .sources import small

LOCAL = ZoneInfo('Europe/Amsterdam')
TEAMS = {'3.01': ('tomatonuts', 'camir_27'), '3.02': ('mugrow', 'cam_21'),
         '3.03': ('agrifusion', 'cam_20'), '3.06': ('reference', 'cam_1'),
         '3.07': ('ideas', 'cam_19'), '3.08': ('trigger', 'cam_18')}
DAYS = [date(2024, 10, 7), date(2024, 10, 22), date(2024, 11, 5)]
DENSITY = 'dwarf_tomato/plant_density'
HARVEST_DAY = 'dwarf_tomato/harvest_date'
START = date(2024, 9, 3)
PREFIX = 'autonomous_greenhouse_challenge4_timeseries/'
SITE = ['compartment/air_temperature', 'compartment/par', 'compartment/relative_humidity',
        'compartment/co2_concentration']
CONTROLS = ['compartment/' + col for col in (
    'heating_lower_circuit/pipe_temperature', 'screen_energy/screen_position',
    'screen_blackout/screen_position', 'water_supply/water_flow_duration',
    'window_position_lee_side', 'window_position_wind_side',
    'co2_actuation_state', 'lamps_activation_percentage')]
WEATHER = ['weather/air_temperature.outside', 'weather/par.outside', 'weather/radiation_global']
ENV = {'inside_measured': SITE, 'actual_control_state': CONTROLS,
       'outside_measured': WEATHER, 'logged_density_asof': [DENSITY]}
IMAGE = re.compile(r'(?P<cam>cam(?:ir)?_\d+)_(?P<stamp>\d{4}(?:_\d{2}){5})_(?P<kind>rgb|depth)\.png$')
PLANT = re.compile(r'(30[123678])-\d+')


def localize(naive):
    candidates = {aware.astimezone(timezone.utc) for fold in (0, 1)
                  if (aware := naive.replace(tzinfo=LOCAL, fold=fold)).astimezone(timezone.utc)
                  .astimezone(LOCAL).replace(tzinfo=None) == naive}
    if len(candidates) != 1:
        raise ValueError(f'ambiguous or nonexistent Amsterdam time: {naive}')
    return next(iter(candidates)).astimezone(LOCAL)


def cutoff(day):
    return datetime.combine(day - timedelta(days=14), time(17), LOCAL)


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_series():
    with zipfile.ZipFile(io.BytesIO(small('series24'))) as z:
        metadata = json.loads(z.read(PREFIX + 'channel_info.json'))
        if metadata['timezone'] != LOCAL.key:
            raise ValueError('Unexpected 2024 time zone')
        columns = set(SITE + CONTROLS + WEATHER + [DENSITY, HARVEST_DAY, 'time'])
        series = {}
        for name in [*(t for t, _ in TEAMS.values()), 'weather', 'weather_forecast']:
            with z.open(PREFIX + f'timeseries/{name}.csv') as f:
                d = pd.read_csv(f, encoding='utf-8-sig', usecols=None if name == 'weather_forecast' else columns.__contains__)
            d['csv_row'] = d.index + 2
            d['utc'] = pd.to_datetime(d.time, utc=True, errors='raise')
            if d.utc.duplicated().any() or not d.utc.is_monotonic_increasing:
                raise ValueError(f'{name}: duplicate or out-of-order time')
            d['local'] = d.utc.dt.tz_convert(LOCAL)
            if not d.time.eq(d.local.dt.strftime('%Y-%m-%d %H:%M:%S%z').str.replace(r'([+-]\d\d)(\d\d)$', r'\1:\2', regex=True)).all():
                raise ValueError(f'{name}: offset mismatch against Amsterdam')
            series[name] = d
        return series, z.read(PREFIX + 'Harvest.xlsx')


def density_events(series):
    rows = []
    for comp, (team, _) in TEAMS.items():
        if team not in series:
            continue
        d = series[team]
        for _, row in d.loc[d[DENSITY].notna()].iterrows():
            value = row[DENSITY]
            known_move = comp == '3.07' and row.local.date() == date(2024,10,21) and value == 30
            rows.append(dict(compartment=comp, team=team, log_timestamp_local=row.local.isoformat(),
                             log_date_local=row.local.date(), utc=row.utc, plants_m2=value,
                             csv_row=row.csv_row, source=f'{PREFIX}timeseries/{team}.csv',
                             valid=number(value) and value > 0, actual_move_timestamp=None,
                             actual_move_date_local=date(2024,10,20) if known_move else None,
                             actual_move_source='doi:10.3390/s25144321 Section 3.3.2' if known_move else None,
                             actual_move_status='date_only_reported_by_staff' if known_move else 'not_verified_no_physical_move_log'))
    return pd.DataFrame(rows)


def asof_density(events, comp, timestamp):
    if events.empty:
        return None
    utc = pd.Timestamp(timestamp).tz_convert('UTC')
    subset = events.loc[(events.compartment == comp) & events.valid & (events.utc <= utc)]
    return None if subset.empty else subset.sort_values('utc').iloc[-1]


def final_dates(series):
    result = {}
    for comp, (team, _) in TEAMS.items():
        if team not in series:
            continue
        d = series[team]
        r = d.loc[d[HARVEST_DAY].notna()]
        if len(r) == 1 and r.iloc[0][HARVEST_DAY] == r.iloc[0].local.dayofyear:
            row = r.iloc[0]
            result[comp] = (row.local.date(), f'{PREFIX}timeseries/{team}.csv:{row.csv_row} [{HARVEST_DAY}={row[HARVEST_DAY]:g}]')
    return result


def harvest(book, finals):
    values = openpyxl.load_workbook(io.BytesIO(book), data_only=True)
    formulas = openpyxl.load_workbook(io.BytesIO(book), data_only=False)
    all_rows = []
    for sheet in values:
        if sheet.title == 'Information':
            continue
        common = sheet.title.startswith('Destructive harvest ')
        if common:
            day = datetime.strptime(sheet.title.split()[-1], '%d-%m-%y').date()
            source = f'worksheet title: {sheet.title}'
            final_comp = None
        else:
            final_comp = re.fullmatch(r'Final Harvest\s+(3\.\d+)\s*', sheet.title, re.I).group(1)
            day, source = finals.get(final_comp, (None, 'unknown_no_verified_harvest_date_event'))
        headers = {str(cell.value).strip().lower(): cell.column for cell in sheet[1] if isinstance(cell.value, str)}
        total = headers.get('fw tot', headers.get('fw'))
        count = next(headers[h] for h in ('# tot', '#tot', '# tom') if h in headers)
        parts = [headers[h] for h in ('fw green', 'fw tom coloured', 'fw coloured', 'fw red', 'fw tom red') if h in headers]
        for i in range(2, sheet.max_row + 1):
            raw = [cell.value for cell in sheet[i]]
            if not any(x is not None for x in raw):
                continue
            plant = raw[headers['plant'] - 1]
            m = PLANT.fullmatch(str(plant))
            weight = raw[total - 1]
            gh = raw[headers['gh #'] - 1] if 'gh #' in headers else None
            formula = formulas[sheet.title].cell(i, total)
            flags = []
            if m:
                comp = f'{int(m[1]) / 100:.2f}'
                kind = 'individual' if number(weight) and weight >= 0 else 'individual_missing_weight'
                if kind != 'individual':
                    flags.append('no_measured_total_fresh_weight')
                if gh is not None and gh != int(m[1]):
                    flags.append('compartment_id_conflict')
                n = raw[count - 1]
                if number(weight) and number(n) and n > 0 and weight/n < 1 and common:
                    flags.append('low_mass_per_fruit_lt_1g_retained')
                component_values = [raw[c - 1] for c in parts]
                if number(weight) and any(number(v) for v in component_values):
                    subtotal = sum(v for v in component_values if number(v))
                    if abs(weight - subtotal) > .05 and all(v is None or number(v) for v in component_values):
                        flags.append('component_sum_disagreement_retained')
                if 'remarks' in headers and raw[headers['remarks'] - 1]:
                    flags.append('source_remark')
            else:
                comp = final_comp
                kind = 'summary' if common else 'final_summary_or_auxiliary'
                if common:
                    declared = gh if gh is not None else plant
                    comp = f'{int(declared) / 100:.2f}' if number(declared) else None
                    if sheet.title == 'Destructive harvest 22-10-24' and i == 43:
                        flags.append('source_307_but_formula_references_308_rows_37_42')
            all_rows.append(dict(compartment=comp, target_date=day, harvest_kind='common' if common else 'final',
                                 row_kind=kind, plant_id=plant if m else None, raw_plant=plant, raw_gh=gh,
                                 raw_total_fresh_weight=weight, fruit_fresh_weight_g_per_plant=weight if kind=='individual' else None,
                                 fruit_count=raw[count - 1], source_file=PREFIX + 'Harvest.xlsx', source_sheet=sheet.title,
                                 source_row=i, source_cell=formula.coordinate, source_header=sheet.cell(1,total).value,
                                 source_formula=formula.value if formula.data_type == 'f' else None,
                                 date_source=source, remarks=raw[headers['remarks']-1] if 'remarks' in headers else None,
                                 quality_flags=';'.join(flags), raw_cells_json=json.dumps(raw, default=str, ensure_ascii=False)))
    ledger = pd.DataFrame(all_rows)
    gt = ledger.loc[ledger.row_kind.eq('individual') & ~ledger.quality_flags.str.contains('compartment_id_conflict')].copy()
    key = ['harvest_kind', 'compartment', 'target_date', 'plant_id']
    duplicated = gt.duplicated(key, keep=False)
    if duplicated.any():
        ledger.loc[gt.index[duplicated], 'quality_flags'] += ';duplicate_within_occasion_excluded'
        gt = gt.loc[~duplicated]
    groups = ['harvest_kind', 'compartment', 'target_date']
    stat = gt.groupby(groups, dropna=False).agg(n=('plant_id', 'size'),
            mean_g_per_plant=('fruit_fresh_weight_g_per_plant', 'mean'),
            sd_g_per_plant=('fruit_fresh_weight_g_per_plant', 'std'), source_sheet=('source_sheet', 'first'),
            source_rows=('source_row', lambda x: ';'.join(map(str,x))), date_source=('date_source','first')).reset_index()
    return gt, stat, ledger


def pairs(names, cameras):
    indexed = defaultdict(lambda: defaultdict(list))
    for name in names:
        m = IMAGE.fullmatch(name.rsplit('/',1)[-1])
        if m and m['cam'] in cameras:
            indexed[m['cam'], m['stamp']][m['kind']].append(name)
    out = []
    for (cam, stamp), members in sorted(indexed.items()):
        naive = datetime.strptime(stamp, '%Y_%m_%d_%H_%M_%S')
        try:
            local = localize(naive)
            utc, status = local.astimezone(timezone.utc), 'ok'
        except ValueError:
            local, utc, status = None, None, 'ambiguous_or_nonexistent_local_time'
        rgb, depth = members['rgb'], members['depth']
        out.append(dict(camera=cam, local_filename_time=naive, timestamp_local=local.isoformat() if local else None,
                        timestamp_utc=utc, hour=naive.hour, local_date=naive.date(), timestamp_status=status,
                        rgb_members=';'.join(rgb), depth_members=';'.join(depth), rgb_n=len(rgb), depth_n=len(depth),
                        paired=len(rgb)==len(depth)==1 and status=='ok'))
    return pd.DataFrame(out)


def gap_audit(index):
    gaps = []
    stats = []
    for comp, (_, cam) in TEAMS.items():
        sub = index[index.camera.eq(cam)]
        if sub.empty:
            continue
        slots = Counter(zip(sub.local_date, sub.hour))
        first, last = min(slots), max(slots)
        for i in range((last[0]-first[0]).days + 1):
            d = first[0] + timedelta(days=i)
            for h in (4, 8, 12, 16):
                if first <= (d,h) <= last and slots[d,h] != 1:
                    gaps.append(dict(compartment=comp, camera=cam, local_date=d, hour=h,
                                     issue='missing_both' if slots[d,h]==0 else 'multiple_timestamps_same_slot', n=slots[d,h]))
        for row in sub.itertuples():
            if not row.paired or row.hour not in (4,8,12,16):
                gaps.append(dict(compartment=comp, camera=cam, local_date=row.local_date, hour=row.hour,
                                 issue='unpaired_duplicate_invalid_or_off_schedule', n=row.rgb_n + row.depth_n))
        stats.append(dict(compartment=comp, camera=cam, first=first, last=last, rgb=int(sub.rgb_n.sum()),
                          depth=int(sub.depth_n.sum()), paired=int(sub.paired.sum()), unpaired=int((~sub.paired).sum()),
                          issues=sum(g['compartment']==comp for g in gaps)))
    return stats, pd.DataFrame(gaps, columns=['compartment','camera','local_date','hour','issue','n'])


def manifest(index, stat, series, events):
    result = []
    for comp, (team, cam) in TEAMS.items():
        for day in DAYS:
            stop = cutoff(day)
            end = datetime.combine(day, time(0), LOCAL)
            start = datetime.combine(START, time(0), LOCAL)
            if index.empty:
                history_n = None
                last_frame = None
                nominal_missing = None
            else:
                sub = index.loc[index.camera.eq(cam) & index.paired & index.timestamp_utc.between(start,stop)]
                history_n = len(sub)
                last_frame = sub.timestamp_utc.max() if history_n else None
                nominal_missing = ((stop.date()-START).days+1)*4 - len(set(zip(sub.local_date, sub.hour)))
            gt = stat.loc[stat.harvest_kind.eq('common') & stat.compartment.eq(comp) & stat.target_date.eq(day)] if not stat.empty else stat
            n = int(gt.iloc[0].n) if len(gt)==1 else 0
            core = {}
            required_ticks = pd.date_range(stop, end, inclusive='left', freq='5min', tz=LOCAL).tz_convert('UTC')
            for name, fields in [(team, SITE+CONTROLS), ('weather', WEATHER)]:
                if name in series:
                    d = series[name].set_index('utc').reindex(required_ticks)
                    for f in fields:
                        core[f] = int(d[f].notna().sum()) if f in d else 0
            den = asof_density(events,comp,stop)
            if den is not None:
                core[DENSITY] = len(required_ticks)
            issues = []
            if not n:
                issues.append('missing_harvest_truth')
            if not history_n:
                issues.append('pair_index_unavailable' if history_n is None else 'no_pre_cutoff_pairs')
            if not all(core.get(f,0)>0 for f in ('compartment/air_temperature','compartment/par')):
                issues.append('missing_core_actual_environment')
            if den is None:
                issues.append('no_valid_density_at_cutoff')
            result.append(dict(sample_id=f'{comp}/{day}', compartment=comp, team=team, camera=cam,
                               target_date_local=day, target_date_source=gt.iloc[0].date_source if n else None,
                               cutoff_local=stop.isoformat(), cutoff_utc=stop.astimezone(timezone.utc), lead_local_days=14,
                               history_start_local=start.isoformat(), history_paired_frames=history_n,
                               history_missing_nominal_slots=nominal_missing, last_pair_utc=last_frame,
                               target_n_plants=n, target_mean_g_per_plant=gt.iloc[0].mean_g_per_plant if n else None,
                               target_sd_g_per_plant=gt.iloc[0].sd_g_per_plant if n else None,
                               target_sheet=gt.iloc[0].source_sheet if n else None, target_rows=gt.iloc[0].source_rows if n else None,
                               holdout_fold=f'holdout_{comp}', test_compartment=comp,
                               train_compartments=';'.join(c for c in TEAMS if c!=comp),
                               environment_protocol='retrospective_given_realized_environment',
                               env_end_local_exclusive=end.isoformat(), env_expected_5min_ticks=len(required_ticks),
                               allowed_post_cutoff_fields=json.dumps(ENV,sort_keys=True),
                               env_nonmissing_ticks_by_field=json.dumps(core,sort_keys=True),
                               density_at_cutoff_plants_m2=den.plants_m2 if den is not None else None,
                               weather_forecast_prospective_eligible=False,
                               weather_forecast_reason='no_verifiable_issue_time',
                               index_eligible=not issues, exclusion_reason=';'.join(issues),
                               rgbd_model_ready=False, rgbd_readiness_reason='absolute_height_calibration_not_validated'))
    return pd.DataFrame(result)
