import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import json
import time

import numpy as np

from geometry import sample_convex_shapes, signed_distance
from solvers.lbm import solve_batch

# --- settings (s = 1.5) ---
SEED = 0
N_SHAPES = 5
s = 1.5
r0 = 10 * s
Ny, Nx = int(100 * s), int(400 * s)
cx, cy = int(100 * s), int(50 * s)
RE, U_IN, N_STEPS = 20, 0.1, 60_000
SHAPE_KW = dict(eps=0.08, ell=1.0, a1_max=0.3, a2_max=0.18, tol=-0.2)


if __name__ == "__main__":
    masks, radii, margins = sample_convex_shapes(
        N_SHAPES, Ny, Nx, cx, cy, r0=r0, rng=np.random.default_rng(SEED), **SHAPE_KW
    )

    t0 = time.perf_counter()
    rho, u, converged, steps = solve_batch(
        masks, D=2 * r0, Re=RE, u_in=U_IN, max_steps=N_STEPS
    )
    print(
        f"solved {N_SHAPES} shapes in {(time.perf_counter() - t0) / 60:.1f} min, "
        f"{steps} steps"
    )

    ux = u[:, 0].astype(np.float32)  # (N, Ny, Nx)
    uy = u[:, 1].astype(np.float32)
    p = (rho / 3).astype(np.float32)
    p[masks] = 0
    sdf = np.array([signed_distance(m) for m in masks]).astype(np.float32)

    keep = converged
    print(f"{(~keep).sum()} of {len(keep)} shapes did not converge, dropping them")
    masks, sdf, ux, uy, p = masks[keep], sdf[keep], ux[keep], uy[keep], p[keep]
    radii, margins = radii[keep], margins[keep]

    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    np.savez_compressed(
        data_dir / "train_convex.npz",
        masks=masks,
        sdf=sdf,
        ux=ux,
        uy=uy,
        p=p,
        radii=radii,
        margins=margins,
    )
    with open(data_dir / "train_convex.json", "w") as fp:
        json.dump(
            dict(
                seed=SEED,
                n=len(ux),
                s=s,
                r0=r0,
                Ny=Ny,
                Nx=Nx,
                cx=cx,
                cy=cy,
                Re=RE,
                u_in=U_IN,
                **SHAPE_KW,
            ),
            fp,
            indent=2,
        )
    print(f"saved {len(ux)} shapes to {data_dir}")
