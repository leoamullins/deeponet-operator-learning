import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
NAMES = ["$u_x$", "$u_y$", "$p$"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="per-shape error vs convexity margin.")
    parser.add_argument("csv", help="output of scripts/evaluate.py")
    parser.add_argument(
        "--out", default=str(ROOT / "docs/figures/error_vs_convexity.png")
    )
    args = parser.parse_args()

    d = np.genfromtxt(args.csv, delimiter=",", names=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for ax, col, name in zip(axes, ["ux", "uy", "p"], NAMES):
        rho, _ = spearmanr(d["margin"], d[col])
        ax.scatter(d["margin"], d[col], s=14, color="#3B7DD8", alpha=0.8, linewidths=0)
        ax.set_title(f"{name}   Spearman $\\rho$ = {rho:.2f}")
        ax.set_xlabel("convexity margin (lower = less convex)")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("relative $L^2$ per shape")

    fig.savefig(args.out, dpi=100)
    print(f"saved {args.out}")
