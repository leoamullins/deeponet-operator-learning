import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

import argparse
import csv

import numpy as np
import torch

from data import EVAL_BAND, downsample_sdf, load_data, rel_l2
from model import DeepONet

BATCH = 32  # shapes per forward pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="per-shape errors of a checkpoint.")
    parser.add_argument("ckpt", help="e.g. checkpoints/best.pt")
    parser.add_argument("dataset", help="name in data/, e.g. train_convex")
    parser.add_argument("--val", action="store_true", help="only the checkpoint's val_idx")
    parser.add_argument("--out", help="CSV path (default: data/<dataset>_eval.csv)")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = DeepONet(**ckpt["model_cfg"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    mean, std = ckpt["mean"], ckpt["std"]

    path = ROOT / "data" / args.dataset
    if args.val:
        trained_on = ckpt["data_cfg"]["data"]
        assert trained_on == str(path.relative_to(ROOT)), (
            f"--val indices refer to {trained_on}, not data/{args.dataset}"
        )
    sdf, masks, fields, coords = load_data(path)
    with np.load(str(path) + ".npz") as d:
        margins = d["margins"]
    N = len(masks)
    idx = np.array(ckpt["data_cfg"]["val_idx"]) if args.val else np.arange(N)

    X = downsample_sdf(sdf[idx], tuple(ckpt["data_cfg"]["sdf_res"]))
    Y = torch.tensor(fields[idx].reshape(len(idx), -1, 3), dtype=torch.float32)
    M = torch.tensor(~masks[idx].reshape(len(idx), -1), dtype=torch.float32)
    Wm = M * torch.tensor(np.abs(sdf[idx].reshape(len(idx), -1)) < EVAL_BAND)
    Q = torch.tensor(coords)

    rel, rel_wall = [], []
    with torch.no_grad():
        for s in range(0, len(idx), BATCH):
            pred = model(X[s : s + BATCH], Q) * std + mean
            rel.append(rel_l2(pred, Y[s : s + BATCH], M[s : s + BATCH]))
            rel_wall.append(rel_l2(pred, Y[s : s + BATCH], Wm[s : s + BATCH]))
    rel, rel_wall = torch.cat(rel).numpy(), torch.cat(rel_wall).numpy()

    out = Path(args.out) if args.out else ROOT / "data" / f"{args.dataset}_eval.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "margin", "ux", "uy", "p", "ux_wall", "uy_wall", "p_wall"])
        for i, r, rw in zip(idx, rel, rel_wall):
            w.writerow([i, margins[i], *r, *rw])

    print(f"{len(idx)} shapes, mean rel L2")
    print(f"  all fluid  ux {rel[:, 0].mean():.3f}  uy {rel[:, 1].mean():.3f}  p {rel[:, 2].mean():.3f}")
    print(f"  wall band  ux {rel_wall[:, 0].mean():.3f}  uy {rel_wall[:, 1].mean():.3f}  p {rel_wall[:, 2].mean():.3f}")
    print(f"saved {out}")
