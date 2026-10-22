import json

import numpy as np
import torch
import torch.nn.functional as F

EVAL_BAND = 0.3  # fixed band for the wall validation error, comparable across runs


def crop(a, meta):
    """Crop (..., Ny, Nx) to [-2 r0, 8 r0] x [-2.5 r0, 2.5 r0] around the body."""
    cx, cy, r0 = meta["cx"], meta["cy"], meta["r0"]
    y0, y1 = cy - int(2.5 * r0), cy + int(2.5 * r0)
    x0, x1 = cx - int(2 * r0), cx + int(8 * r0)
    return a[..., y0:y1, x0:x1]


def crop_fields(ux, uy, p, meta):
    """Cropped (N, H, W, 3) fields, with pressure relative to the upstream edge."""
    ux, uy, p = crop(ux, meta), crop(uy, meta), crop(p, meta)
    p = p - p[:, :, :1].mean(axis=(1, 2), keepdims=True)
    return np.stack([ux, uy, p], axis=-1)


def grid_coords(meta):
    """(H * W, 2) query points of the crop, in units of r0 from the body centre."""
    cx, cy, r0 = meta["cx"], meta["cy"], meta["r0"]
    yy, xx = crop(np.mgrid[: meta["Ny"], : meta["Nx"]], meta)
    coords = np.stack([(xx - cx) / r0, (yy - cy) / r0], axis=-1).reshape(-1, 2)
    return coords.astype(np.float32)


def load_data(path):
    with open(str(path) + ".json") as f:
        meta = json.load(f)

    with np.load(str(path) + ".npz") as d:
        masks = crop(d["masks"], meta)
        sdf = crop(d["sdf"], meta) / meta["r0"]  # sdf in units of r0
        fields = crop_fields(d["ux"], d["uy"], d["p"], meta)

    return sdf, masks, fields, grid_coords(meta)


def downsample_sdf(sdf, res):
    """(N, H, W) SDF -> (N, res[0] * res[1]) branch input."""
    sdf_t = torch.tensor(sdf, dtype=torch.float32).unsqueeze(1)  # (N, 1, H, W)
    return F.interpolate(
        sdf_t, size=res, mode="bilinear", align_corners=False
    ).flatten(1)


def rel_l2(pred, true, mask):
    """Per-shape relative L2 over masked cells, (B, 3)."""
    m = mask[..., None]
    num = (((pred - true) * m) ** 2).sum(dim=1).sqrt()
    den = ((true * m) ** 2).sum(dim=1).sqrt()
    return num / den
