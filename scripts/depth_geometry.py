"""Camera-coordinate geometry for the Phase 0.5 diagnostics."""

import numpy as np


def valid_depth(raw):
    # Zero is the SDK invalid code; all-ones is an additional audit guard.
    return (raw > 0) & (raw < 65535)


def project(u, v, raw, intr, scale):
    z = np.asarray(raw, dtype=float) * scale
    return np.stack((z * (u - intr['ppx']) / intr['fx'],
                     z * (v - intr['ppy']) / intr['fy'], z), axis=-1)


def fit_plane(points):
    if len(points) < 30:
        return None
    # Report residuals on all input points, even those excluded from the refit.
    keep = np.ones(len(points), dtype=bool)
    for iteration in range(4):
        if keep.sum() < 30:
            return None
        p = points[keep]
        center = p.mean(axis=0)
        _, spread, vectors = np.linalg.svd(p - center, full_matrices=False)
        if spread[1] < 1e-6:
            return None
        n = vectors[-1] * (-1 if vectors[-1, 2] > 0 else 1)
        d = -n @ center
        residual = points @ n + d
        if iteration < 3:
            middle = np.median(residual)
            mad = np.median(np.abs(residual - middle))
            keep = np.abs(residual - middle) <= max(.006, 3 * 1.4826 * mad)
    return n, float(d), residual, keep


def plane_intersection(u, v, intr, n, d):
    ray = project(np.asarray(u), np.asarray(v), np.ones_like(u), intr, 1.)
    return ray * (-d / (ray @ n))[..., None]


def canopy_h90(raw, mask, intr, scale, plane, *, scale_passed, plane_passed):
    if not (scale_passed and plane_passed and plane is not None):
        return None
    v, u = np.nonzero(mask & valid_depth(raw))
    if not len(v):
        return None
    n, d = plane
    return float(np.percentile(project(u, v, raw[v, u], intr, scale) @ n + d, 90))


def candidate_points(raw, mask, boxes, intr, scale):
    """Per-region samples. Never use RGB canopy-selected pixels in plane fits."""
    for (x0, y0, x1, y1) in boxes:
        roi = raw[y0:y1:3, x0:x1:3]
        v, u = np.nonzero(valid_depth(roi) & ~mask[y0:y1:3, x0:x1:3])
        yield project(x0 + 3*u, y0 + 3*v, roi[v, u], intr, scale)


def summarize_plane(points_by_patch):
    points = np.concatenate(points_by_patch)
    fitted = fit_plane(points)
    if fitted is None:
        return {'n_points': len(points), 'status': 'insufficient_or_degenerate_points'}
    n, d, residual, keep = fitted
    return dict(status='diagnostic_fit_only', normal=n.tolist(), d_m=d,
                n_points=len(points), n_inliers=int(keep.sum()),
                rmse_mm=float(np.sqrt(np.mean(residual**2))*1000),
                p90_abs_mm=float(np.percentile(np.abs(residual), 90)*1000),
                inlier_rmse_mm=float(np.sqrt(np.mean(residual[keep]**2))*1000),
                patch_counts=[len(p) for p in points_by_patch],
                patch_signed_median_mm=[float(np.median(p @ n + d)*1000) if len(p) else None
                                        for p in points_by_patch])


def edge_probe(raw, x, y):
    """Compare a manually marked RGB rim edge to the largest local axial-Z jump."""
    radius, half_step = 48, 4
    roi = raw[y-radius-half_step:y+radius+half_step+1, x-3:x+4]
    if roi.shape != (2*(radius+half_step)+1, 7):
        return {'status': 'probe_outside_image'}
    z = np.median(roi, axis=1)
    ok = (valid_depth(z[:-2*half_step]) & valid_depth(z[2*half_step:]))
    if not ok.any():
        return {'status': 'no_valid_depth_edge'}
    # Restrict to the expected direction: the exterior bench lies farther away.
    dz = np.where(ok, z[2*half_step:] - z[:-2*half_step], -np.inf)
    i = int(dz.argmax())
    return dict(status='positive_depth_transition' if dz[i] > 0 else 'no_positive_transition',
                offset_y_px=i-radius if dz[i] > 0 else None, depth_jump_raw=float(dz[i]),
                search_halfwidth_px=radius, compared_row_separation_px=2*half_step)
