import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import ConvexHull

from grf import build_cholesky_factor, sample_functions

N_THETA = 128
THETA = np.linspace(0, 2 * np.pi, N_THETA, endpoint=False)


def sample_radii(n_shapes, r0, eps, ell, rng, a1_max=0.3, a2_max=0.25):
    L = build_cholesky_factor(THETA, ell, periodic=True)
    g = sample_functions(L, n_shapes, rng)
    a1 = rng.uniform(0, a1_max, size=(n_shapes, 1))
    a2 = rng.uniform(0, a2_max, size=(n_shapes, 1))
    phi1 = rng.uniform(0, 2 * np.pi, size=(n_shapes, 1))
    phi2 = rng.uniform(0, 2 * np.pi, size=(n_shapes, 1))
    radii = r0 * (
        1 + a1 * np.cos(THETA - phi1) + a2 * np.cos(2 * (THETA - phi2)) + eps * g
    )
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


def sample_convex_shapes(
    n_shapes, Ny, Nx, cx, cy, r0, eps, ell, rng, a1_max=0.3, a2_max=0.18, tol=-0.2,
    max_rounds=100,
):
    kept_radii, kept_margins = [], []
    for _ in range(max_rounds):
        if len(kept_radii) >= n_shapes:
            break
        radii = sample_radii(n_shapes, r0, eps, ell, rng, a1_max=a1_max, a2_max=a2_max)
        convex, margin = is_convex(radii, tol=tol)
        kept_radii.extend(radii[convex])
        kept_margins.extend(margin[convex])
    if len(kept_radii) < n_shapes:
        raise RuntimeError(
            f"only {len(kept_radii)}/{n_shapes} convex shapes after {max_rounds} rounds; loosen tol or amplitudes"
        )

    radii = np.array(kept_radii[:n_shapes])
    margins = np.array(kept_margins[:n_shapes])
    masks = np.array([shape_mask(r, Ny, Nx, cx, cy) for r in radii])
    return masks, radii, margins


def sample_lobed_shapes(
    n_shapes, Ny, Nx, cx, cy, r0, rng, amp=(0.1, 0.4), lobes=(3, 4, 5)
):
    """r = r0 (1 + a cos k(theta - phi)): concave once a is large enough for the lobe count k."""
    k = rng.choice(lobes, size=(n_shapes, 1))
    a = rng.uniform(*amp, size=(n_shapes, 1))
    phi = rng.uniform(0, 2 * np.pi, size=(n_shapes, 1))
    radii = r0 * (1 + a * np.cos(k * (THETA - phi)))
    _, margins = is_convex(radii)
    masks = np.array([shape_mask(r, Ny, Nx, cx, cy) for r in radii])
    return masks, radii, margins


def is_convex(radii, tol=0.0):
    N = radii.shape[-1]
    k = np.fft.fftfreq(N, d=1 / N)  # integer wavenumbers, since theta has period 2pi
    r_hat = np.fft.fft(radii, axis=-1)

    k1 = k.copy()
    k1[N // 2] = 0  # drop the Nyquist mode for the odd derivative
    r1 = np.fft.ifft(1j * k1 * r_hat, axis=-1).real  # dr/dtheta
    r2 = np.fft.ifft(-(k**2) * r_hat, axis=-1).real  # d2r/dtheta2

    curv = radii**2 + 2 * r1**2 - radii * r2
    margin = (curv / radii**2).min(axis=-1)  # dimensionless; 1.0 for a perfect circle
    return margin >= tol, margin
