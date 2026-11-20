import json
import unittest
from datetime import date, datetime, timedelta

import pandas as pd

from scripts.sources import ROOT
from scripts.trial import TEAMS, asof_density, cutoff, localize, pairs

AUDIT = ROOT / 'audit'


class TimeAndIndexTests(unittest.TestCase):
    def test_dst_disambiguation(self):
        self.assertEqual(localize(datetime(2024, 10, 26, 4)).utcoffset(), timedelta(hours=2))
        self.assertEqual(localize(datetime(2024, 10, 27, 4)).utcoffset(), timedelta(hours=1))
        for dt in [datetime(2024, 3, 31, 2, 2), datetime(2024, 10, 27, 2, 2)]:
            with self.assertRaises(ValueError):
                localize(dt)

    def test_exact_pairing_rejects_duplicate_and_nearby_capture(self):
        stem = 'cam_1_2024_09_23_12_00_00'
        files = [f'{stem}_rgb.png', f'{stem}_depth.png']
        self.assertTrue(pairs(files, {'cam_1'}).paired.item())
        self.assertFalse(pairs(files + files[:1], {'cam_1'}).paired.any())
        self.assertFalse(pairs([files[0], files[1].replace('12_00_00','12_00_01')], {'cam_1'}).paired.any())

    def test_density_never_looks_forward(self):
        events = pd.read_csv(AUDIT / 'density_events.csv',dtype={'compartment':str})
        events.utc = pd.to_datetime(events.utc,utc=True)
        self.assertIsNone(asof_density(events,'3.06',localize(datetime(2024,9,2,17))))
        self.assertEqual(asof_density(events,'3.06',localize(datetime(2024,10,6,23,59))).plants_m2,30)
        self.assertEqual(asof_density(events,'3.06',localize(datetime(2024,10,7,0))).plants_m2,20)
        date_only = events.loc[events.actual_move_status.eq('date_only_reported_by_staff')].iloc[0]
        self.assertEqual(date_only.actual_move_date_local,'2024-10-20')
        self.assertTrue(pd.isna(date_only.actual_move_timestamp))


class PublishedAuditTests(unittest.TestCase):
    def test_weights_and_sources(self):
        gt = pd.read_csv(AUDIT / 'harvest_gt.csv',dtype={'compartment':str})
        common = gt.loc[gt.harvest_kind.eq('common')]
        self.assertEqual(len(common),108)
        self.assertEqual(len(gt)-len(common),171)
        self.assertTrue(gt.fruit_fresh_weight_g_per_plant.notna().all())
        self.assertEqual(len(common.groupby(['compartment','target_date'])),18)
        suspicious = common.loc[common.quality_flags.str.contains('low_mass',na=False)]
        self.assertEqual(suspicious.fruit_fresh_weight_g_per_plant.item(),19.62)
        self.assertTrue(gt.date_source.notna().all())

    def test_manifest_can_be_rejoined_without_leakage(self):
        m = pd.read_csv(AUDIT / 'forecast_manifest.csv',dtype={'compartment':str,'test_compartment':str})
        i = pd.read_csv(AUDIT / 'frame_index.csv')
        i.timestamp_utc = pd.to_datetime(i.timestamp_utc,utc=True)
        self.assertEqual(len(m),18)
        self.assertTrue(m.sample_id.is_unique and m.index_eligible.all())
        self.assertFalse(m.rgbd_model_ready.any() or m.weather_forecast_prospective_eligible.any())
        for r in m.itertuples():
            self.assertEqual(r.cutoff_local,cutoff(date.fromisoformat(r.target_date_local)).isoformat())
            f = i.loc[i.camera.eq(r.camera)& i.paired & i.timestamp_utc.between(pd.Timestamp(r.history_start_local),pd.Timestamp(r.cutoff_utc))]
            self.assertEqual(len(f),r.history_paired_frames)
            self.assertEqual(f.timestamp_utc.max(),pd.Timestamp(r.last_pair_utc))
            self.assertEqual(set(r.train_compartments.split(';')),set(TEAMS)-{r.test_compartment})
            self.assertEqual(r.test_compartment,r.compartment)
            self.assertNotIn('harvest_date',r.allowed_post_cutoff_fields)

    def test_reference_samples_and_no_fabricated_height(self):
        obs = pd.read_parquet(AUDIT / 'compartment_obs.parquet')
        self.assertEqual(len(obs),6)
        self.assertTrue(obs.rgb_coverage_fraction.between(0,1).all())
        self.assertTrue(obs.mask_valid_depth_fraction.between(0,1).all())
        self.assertTrue(obs.h90_m.isna().all() and obs.h90_missing_reason.notna().all())
        self.assertTrue((obs.roi == '0,0,3840,2160').all())
        j = pd.read_csv(AUDIT / 'reference_cutoff_join.csv')
        self.assertEqual(len(j),3)
        self.assertTrue((pd.to_datetime(j.timestamp_utc,utc=True) <= pd.to_datetime(j.cutoff_local,utc=True)).all())

    def test_pretrial_provenance_and_conflict(self):
        s = pd.read_csv(AUDIT / 'pretrial_single_join.csv')
        c = pd.read_csv(AUDIT / 'pretrial_canopy_join.csv')
        self.assertEqual(s.paired_label.sum(),479)
        self.assertEqual(s.decoded.sum(),4)
        self.assertEqual(s.destructive_candidates.eq(1).sum(),143)
        conflict = s.loc[s.destructive_trait_checks.map(lambda x:json.loads(x)['leaf_area']=='different')]
        self.assertEqual(conflict.leaf_area.item(),4287.85)
        self.assertEqual(conflict.workbook_leaf_area.item(),1287.85)
        self.assertEqual(c.decoded.sum(),3)
        self.assertEqual(c.reliable_plant_or_roi_links.sum(),0)
