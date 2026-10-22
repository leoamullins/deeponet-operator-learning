import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

import argparse
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm

from data import crop, crop_fields, downsample_sdf, grid_coords
from geometry import THETA, is_convex, shape_mask, signed_distance
from model import DeepONet
from solvers.lbm import solve_batch

# trefoil: r = r0 (1 + a cos k(theta - phi)); k >= 3 with a large enough is non-convex
K_LOBES, AMP, PHI = 3, 0.3, 0.0
NAMES = ["$u_x$", "$u_y$", "$p$"]


def show(ax, img, solid, extent, **kw):
    im = ax.imshow(np.ma.masked_where(solid, img), extent=extent, origin="lower", **kw)
    ax.imshow(
        np.ma.masked_where(~solid, solid),
        extent=extent,
        origin="lower",
        cmap="Greys",
        vmin=0,
        vmax=1.5,
    )
    ax.set_aspect("equal")
    return im


def plot_fields(t3, p3, solid, extent, title, rel):
    """t3, p3: (H, W, 3) truth and prediction; solid: (H, W) body mask."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 6.5), constrained_layout=True)
    for c, name in enumerate(NAMES):
        t, p_ = t3[..., c], p3[..., c]
        e = np.abs(p_ - t)
        lo, hi = t[~solid].min(), t[~solid].max()
        if c == 0:  # ux: magnitude
            kw = dict(cmap="viridis", vmin=lo, vmax=hi)
        else:  # uy, p: signed, centre the colormap at 0
            kw = dict(
                cmap="RdBu_r", norm=TwoSlopeNorm(0, min(lo, -1e-9), max(hi, 1e-9))
            )
        im = show(axes[c, 0], t, solid, extent, **kw)
        show(axes[c, 1], p_, solid, extent, **kw)
        fig.colorbar(im, ax=axes[c, :2], shrink=0.9)
        im = show(axes[c, 2], e, solid, extent, cmap="Purples", vmin=0, vmax=e[~solid].max())
        fig.colorbar(im, ax=axes[c, 2], shrink=0.9)
        axes[c, 0].set_ylabel(f"{name}\n$y / r_0$")
        axes[c, 2].set_title(f"|error|  (rel L2 {rel[c]:.3f})", fontsize=10)
    axes[0, 0].set_title("LBM (truth)")
    axes[0, 1].set_title("DeepONet")
    for ax in axes[-1]:
        ax.set_xlabel("$x / r_0$")
    fig.suptitle(title)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepONet vs LBM on a non-convex trefoil.")
    parser.add_argument("ckpt", help="e.g. checkpoints/best.pt")
    parser.add_argument("--out", default=str(ROOT / "docs/figures/nonconvex_trefoil.png"))
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = DeepONet(**ckpt["model_cfg"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    cfg = ckpt["data_cfg"]
    with open(ROOT / (cfg["data"] + ".json")) as f:
        meta = json.load(f)  # grid and flow settings of the training data
    r0, Ny, Nx = meta["r0"], meta["Ny"], meta["Nx"]

    radii = r0 * (1 + AMP * np.cos(K_LOBES * (THETA - PHI)))
    _, margin = is_convex(radii[None])
    print(f"convexity margin {margin[0]:.2f} (training shapes: >= {meta.get('tol')})")
    mask = shape_mask(radii, Ny, Nx, meta["cx"], meta["cy"])
    solid = crop(mask, meta)

    coords = grid_coords(meta)
    x = downsample_sdf(crop(signed_distance(mask), meta)[None] / r0, tuple(cfg["sdf_res"]))
    with torch.no_grad():
        pred = model(x, torch.tensor(coords)) * ckpt["std"] + ckpt["mean"]
    pred = pred.numpy().reshape(*solid.shape, 3)

    # ground truth with the LBM, same settings as the dataset
    t0 = time.perf_counter()
    rho, u, conv, steps = solve_batch(
        mask[None], D=2 * r0, Re=meta["Re"], u_in=meta["u_in"], max_steps=60_000
    )
    print(f"LBM converged {conv[0]} after {steps} steps, {time.perf_counter() - t0:.0f} s")
    if not conv[0]:
        raise RuntimeError("LBM did not converge; ground truth is unreliable")
    p = rho / 3
    p[:, mask] = 0
    true = crop_fields(u[:, 0], u[:, 1], p, meta)[0]

    fluid = ~solid
    rel = np.linalg.norm((pred - true)[fluid], axis=0) / np.linalg.norm(true[fluid], axis=0)
    print("rel L2 (ux, uy, p):", rel.round(3))

    extent = [coords[:, 0].min(), coords[:, 0].max(), coords[:, 1].min(), coords[:, 1].max()]
    fig = plot_fields(true, pred, solid, extent, f"non-convex shape, margin {margin[0]:.2f}", rel)
    fig.savefig(args.out, dpi=100)
    print(f"saved {args.out}")
