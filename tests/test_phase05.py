from io import StringIO
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.depth_geometry import candidate_points, canopy_h90, fit_plane, plane_intersection, project, valid_depth
from scripts.phase05 import BASE_DIR, coverage_table, load_scope


class DepthGeometryTests(unittest.TestCase):
    def test_millimeters_to_rgb_optical_coordinates(self):
        k = dict(fx=500, fy=1000, ppx=320, ppy=240)
        point = project(np.array([420]), np.array([340]), np.array([1500]), k, .001)
        np.testing.assert_allclose(point, [[.3, .15, 1.5]])
        self.assertEqual(valid_depth(np.array([0,1,65534,65535])).tolist(), [False, True, True, False])
        np.testing.assert_allclose(plane_intersection(420, 340, k, np.array([0,0,-1]), 1.5), point[0])

    def test_tilted_plane_and_signed_h90(self):
        u, v = np.meshgrid(np.arange(8), np.arange(8))
        z = .2*u + 2
        normal, offset, residual, _ = fit_plane(np.column_stack([u.ravel(), v.ravel(), z.ravel()]))
        np.testing.assert_allclose(normal, np.array([.2,0,-1])/np.sqrt(1.04), atol=1e-10)
        np.testing.assert_allclose(residual, 0, atol=1e-10)
        raw = np.array([[1800, 1600, 65535, 1400]], dtype=np.uint16)
        mask = np.array([[True, True, True, False]])
        k = dict(fx=10, fy=10, ppx=0, ppy=0)
        expected = np.percentile(offset + np.array([-1.8, .2*.16-1.6])/np.sqrt(1.04), 90)
        self.assertAlmostEqual(canopy_h90(raw, mask, k, .001, (normal, offset),
                                         scale_passed=True, plane_passed=True), expected)
        for scale_ok, plane_ok in [(True, False), (False, True), (False, False)]:
            self.assertIsNone(canopy_h90(raw, mask, k, .001, (normal, offset),
                                         scale_passed=scale_ok, plane_passed=plane_ok))

    def test_reference_fit_disallows_masked_depth(self):
        raw = np.full((9,9), 1400, dtype=np.uint16)
        raw[3,3] = 900
        mask = raw == 900
        k = dict(fx=10, fy=10, ppx=0, ppy=0)
        points, = candidate_points(raw, mask, [(0,0,9,9)], k, .001)
        self.assertEqual(len(points), 8)
        np.testing.assert_allclose(points[:,2], 1.4)
        self.assertIsNone(fit_plane(points))


class HistoryTests(unittest.TestCase):
    def test_baseline_and_unique_history_union(self):
        frames, manifest = load_scope(BASE_DIR)
        self.assertEqual(len(manifest), 18)
        self.assertEqual(len(frames), 1200)
        self.assertTrue(frames.frame_id.is_unique)
        self.assertEqual(frames.groupby('compartment').size().tolist(), [200]*6)
        ref_changes = frames[(frames.compartment == '3.06') & frames.logged_density_change_day]
        self.assertEqual(ref_changes.groupby('local_date').size().to_dict(),
                         {'2024-09-11': 4, '2024-09-19': 4, '2024-10-07': 4})
        self.assertEqual(frames[frames.documented_move_day].compartment.unique().tolist(), ['3.07'])

    def test_coverage_filters_and_does_not_treat_common_screen_as_h90(self):
        frame = dict(camera='cam_1', paired=True, rgb_decoded=True, depth_decoded=True,
                     attempted=True, rgb_screen_ok=True, common_screen_ok=True, h90_qualified=False)
        q = pd.DataFrame([dict(frame, timestamp_utc=pd.Timestamp(t, tz='UTC'), **extra) for t, extra in [
            ('2024-09-03 12:00', {}), ('2024-10-08 15:00', {}),
            ('2024-10-08 15:01', {}), ('2024-10-06 18:00', {})]])
        q.loc[3,'camera'] = 'cam_18'
        target = SimpleNamespace(camera='cam_1', cutoff_utc=pd.Timestamp('2024-10-08 15:00', tz='UTC'),
                                 history_start_local='2024-09-03T00:00:00+02:00', index_eligible=True)
        row = coverage_table(q, pd.DataFrame([vars(target)])).iloc[0]
        self.assertEqual(row.common_rgbd_frames, 2)
        self.assertEqual(row.available_rgb_frames, 2)
        self.assertEqual(row.qualified_h90_frames, 0)
        self.assertFalse(row.paired_evaluation_eligible)


class PublishedDepthAuditTests(unittest.TestCase):
    def test_published_coverage_matches_frame_quality(self):
        path = BASE_DIR / 'frame_quality.parquet'
        if not path.exists():
            self.skipTest('Run scripts.phase05 to build the published outputs')
        q = pd.read_parquet(path)
        _, manifest = load_scope(BASE_DIR)
        expected = coverage_table(q, manifest)
        pd.testing.assert_frame_equal(pd.read_csv(BASE_DIR/'forecast_coverage.csv'),
                                      pd.read_csv(StringIO(expected.to_csv(index=False))))
        self.assertEqual(len(q), 1200)
        self.assertTrue(q.frame_id.is_unique)
        self.assertTrue((~q.common_screen_ok | (q.rgb_screen_ok & q.paired & q.depth_decoded)).all())
        self.assertTrue(q.loc[~q.h90_qualified, 'h90_m'].isna().all())
        self.assertTrue(q.loc[q.h90_qualified, 'h90_m'].notna().all())
        self.assertTrue(q.loc[~q.h90_qualified, 'failure_reasons'].notna().all())
