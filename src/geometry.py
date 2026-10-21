import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import ConvexHull

from grf import build_cholesky_factor, sample_functions

N_THETA = 128
THETA = np.linspace(0, 2 * np.pi, N_THETA, endpoint=False)


def sample_radii(n_shapes, r0, eps, ell, rng):
    L = build_cholesky_factor(THETA, ell, periodic=True)
    g = sample_functions(L, n_shapes, rng)
    radii = r0 * (1 + eps * g)
    if np.any(radii <= 0):
        raise ValueError("Non-positive radius: reduce eps")
    return radii


def shape_mask(radii, Ny, Nx, cx, cy):
    y, x = np.mgrid[0:Ny, 0:Nx]
    dx, dy = x - cx, y - cy
    angle = np.arctan2(dy, dx) % (2 * np.pi)  # map to [0, 2π)
    dist = np.hypot(dx, dy)

    r_boundary = np.interp(angle, THETA, radii, period=2 * np.pi)
    return dist <= r_boundary


def signed_distance(mask):
    return distance_transform_edt(~mask) - distance_transform_edt(mask)


def convexity_ratio(mask):
    pts = np.argwhere(mask)
    hull = ConvexHull(pts)
    return mask.sum() / hull.volume


def sample_convex_shapes(n_shapes, Ny, Nx, cx, cy, r0, eps, ell, rng, min_ratio=1.0):

    masks, kept = [], []
    while len(masks) < n_shapes:
        for r in sample_radii(n_shapes, r0, eps, ell, rng):
            mask = shape_mask(r, Ny, Nx, cx, cy)
            if convexity_ratio(mask) >= min_ratio:
                masks.append(mask)
                kept.append(r)
            if len(masks) == n_shapes:
                break
    return np.array(masks), np.array(kept)
