import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import argparse
import json
import time

import numpy as np

from geometry import sample_convex_shapes, signed_distance
from solvers.lbm import solve_batch

# --- settings (s = 1.5) ---
SEED = 0
N_SHAPES = 1200
BATCH = 100  # shapes per solve_batch call, keeps GPU memory bounded
s = 1.5
r0 = 10 * s
Ny, Nx = int(100 * s), int(400 * s)
cx, cy = int(100 * s), int(50 * s)
RE, U_IN, N_STEPS = 20, 0.1, 60_000
SHAPE_KW = {"eps": 0.08, "ell": 1.0, "a1_max": 0.3, "a2_max": 0.18, "tol": -0.2}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="generate the convex-shape flow dataset."
    )
    parser.add_argument("--n-shapes", type=int, default=N_SHAPES)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default="train_convex")
    args = parser.parse_args()

    masks, radii, margins = sample_convex_shapes(
        args.n_shapes,
        Ny,
        Nx,
        cx,
        cy,
        r0=r0,
        rng=np.random.default_rng(args.seed),
        **SHAPE_KW,
    )

    t0 = time.perf_counter()
    ux, uy, p, converged = [], [], [], []
    for start in range(0, args.n_shapes, args.batch):
        batch = masks[start : start + args.batch]
        rho, u, conv, steps = solve_batch(
            batch, D=2 * r0, Re=RE, u_in=U_IN, max_steps=N_STEPS
        )
        p_b = (rho / 3).astype(np.float32)
        p_b[batch] = 0
        ux.append(u[:, 0].astype(np.float32))  # (B, Ny, Nx)
        uy.append(u[:, 1].astype(np.float32))
        p.append(p_b)
        converged.append(conv)
        print(
            f"shapes {start}-{start + len(batch) - 1}: {steps} steps, "
            f"{(time.perf_counter() - t0) / 60:.1f} min elapsed",
            flush=True,
        )

    ux, uy, p = np.concatenate(ux), np.concatenate(uy), np.concatenate(p)
    converged = np.concatenate(converged)
    sdf = np.array([signed_distance(m) for m in masks]).astype(np.float32)

    keep = converged
    print(f"{(~keep).sum()} of {len(keep)} shapes did not converge, dropping them")
    masks, sdf, ux, uy, p = masks[keep], sdf[keep], ux[keep], uy[keep], p[keep]
    radii, margins = radii[keep], margins[keep]

    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    np.savez_compressed(
        data_dir / f"{args.out}.npz",
        masks=masks,
        sdf=sdf,
        ux=ux,
        uy=uy,
        p=p,
        radii=radii,
        margins=margins,
    )
    with open(data_dir / f"{args.out}.json", "w") as fp:
        json.dump(
            dict(
                seed=args.seed,
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
