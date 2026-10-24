import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from data import EVAL_BAND, downsample_sdf, load_data, rel_l2
from model import DeepONet

DATA = ROOT / "data" / "train_convex"
OUT = ROOT / "checkpoints"
SEED = 0
N_VAL = 200
SDF_RES = (32, 64)  # branch input resolution (downsampled SDF)
BATCH = 32  # shapes per batch
N_POINTS = 2048  # random query points per batch
EPOCHS = int(os.environ.get("EPOCHS", 50))  # override with EPOCHS=... env var
LR, WEIGHT_DECAY, WARMUP_FRAC = 5e-4, 1e-4, 0.02
P_DIM, WIDTH = 128, 256
# near-wall query sampling: WALL_FRAC of each batch's points come from cells with
# |sdf| < WALL_BAND (in r0) for some training shape; WALL_FRAC=0 is uniform sampling
WALL_FRAC = float(os.environ.get("WALL_FRAC", 0.0))
WALL_BAND = float(os.environ.get("WALL_BAND", 0.3))

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class FlowDataset(Dataset):
    def __init__(self, X, Y, M, Wm):
        self.X, self.Y, self.M, self.Wm = X, Y, M, Wm

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i], self.M[i], self.Wm[i]


def masked_mse(pred, target, fluid):
    err = ((pred - target) ** 2).mean(dim=-1)
    return (err * fluid).sum() / fluid.sum()


@torch.no_grad()
def evaluate(model, loader, Q, mean, std):
    """Mean relative L2 per field over all fluid cells and over the wall band."""
    model.eval()
    rel_sum, wall_sum, n = torch.zeros(3, device=device), torch.zeros(3, device=device), 0
    for xb, yb, mb, wb in loader:
        xb, yb, mb, wb = xb.to(device), yb.to(device), mb.to(device), wb.to(device)
        pred = model(xb, Q) * std + mean
        true = yb * std + mean
        rel_sum += rel_l2(pred, true, mb).sum(dim=0)
        wall_sum += rel_l2(pred, true, wb).sum(dim=0)
        n += len(xb)
    return (rel_sum / n).cpu().numpy(), (wall_sum / n).cpu().numpy()


if __name__ == "__main__":
    torch.manual_seed(SEED)
    sdf, masks, fields, coords = load_data(DATA)
    N, H, Wd = masks.shape
    print(f"{N} shapes, crop {H} x {Wd}, device {device}")

    # train/val splits
    perm = np.random.default_rng(SEED).permutation(N)
    val_idx, tr_idx = perm[:N_VAL], perm[N_VAL:]

    # normalise with training
    fluid_tr = ~masks[tr_idx]
    f_tr = fields[tr_idx][fluid_tr]
    mean = f_tr.mean(0)
    std = f_tr.std(0)
    fields = (fields - mean) / std

    # tensors
    X = downsample_sdf(sdf, SDF_RES)
    Y = torch.tensor(fields.reshape(N, H * Wd, 3), dtype=torch.float32)
    M = torch.tensor(~masks.reshape(N, -1), dtype=torch.float32)
    Wm = M * torch.tensor(np.abs(sdf.reshape(N, -1)) < EVAL_BAND, dtype=torch.float32)

    # cells within WALL_BAND of any training body: the union of the wall layers
    near = (np.abs(sdf[tr_idx]) < WALL_BAND).any(0).reshape(-1)
    near_idx = torch.tensor(np.flatnonzero(near), device=device)
    n_wall = int(WALL_FRAC * N_POINTS)
    print(f"wall sampling: {n_wall}/{N_POINTS} pts from |sdf| < {WALL_BAND} ({near.mean():.1%} of crop)")

    Q = torch.tensor(coords, device=device)  # query grid
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(std, dtype=torch.float32, device=device)

    # datasets and loader
    full_ds = FlowDataset(X, Y, M, Wm)
    train_ds = Subset(full_ds, tr_idx.tolist())
    val_ds = Subset(full_ds, val_idx.tolist())
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=BATCH, shuffle=True, drop_last=True, pin_memory=pin
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, pin_memory=pin)

    # model + optim
    model_cfg: dict[str, Any] = {
        "m": X.shape[1],
        "p": P_DIM,
        "width": WIDTH,
        "n_out": 3,
        "coord_dim": 2,
    }
    model = DeepONet(**model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    total = EPOCHS * len(train_loader)
    warm = int(WARMUP_FRAC * total)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [
            torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warm),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=total - warm, eta_min=LR * 1e-3
            ),
        ],
        milestones=[warm],
    )

    # train
    OUT.mkdir(exist_ok=True)
    best, history = float("inf"), []
    t0 = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb, mb, _ in train_loader:
            opt.zero_grad()
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)

            # shared pts for the batch: uniform + near-wall
            pts = torch.cat(
                [
                    torch.randint(0, H * Wd, (N_POINTS - n_wall,), device=device),
                    near_idx[torch.randint(0, len(near_idx), (n_wall,), device=device)],
                ]
            )
            pred = model(xb, Q[pts])
            loss = masked_mse(pred, yb[:, pts], mb[:, pts])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            epoch_loss += loss.item() / len(train_loader)

        if epoch % 10 == 0 or epoch == EPOCHS:
            rel, rel_wall = evaluate(model, val_loader, Q, mean_t, std_t)
            lr_now = sched.get_last_lr()[0]
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": epoch_loss,
                    "val_rel": rel.tolist(),
                    "val_rel_wall": rel_wall.tolist(),
                    "lr": lr_now,
                }
            )
            print(
                f"epoch {epoch:4d}  loss {epoch_loss:.2e}  "
                f"val rel L2  ux {rel[0]:.3f}  uy {rel[1]:.3f}  p {rel[2]:.3f}  "
                f"wall {rel_wall[0]:.3f} {rel_wall[1]:.3f} {rel_wall[2]:.3f}  "
                f"lr {lr_now:.1e}  ({(time.perf_counter() - t0) / 60:.1f} min)"
            )

            if rel.mean() < best:
                best = rel.mean()
                torch.save(
                    {
                        "model": model.state_dict(),
                        "model_cfg": model_cfg,  # rebuild with DeepONet(**model_cfg)
                        "mean": mean_t.cpu(),
                        "std": std_t.cpu(),
                        "data_cfg": {
                            "data": str(DATA.relative_to(ROOT)),
                            "sdf_res": SDF_RES,
                            "wall_frac": WALL_FRAC,
                            "wall_band": WALL_BAND,
                            "tr_idx": tr_idx.tolist(),
                            "val_idx": val_idx.tolist(),
                        },
                        "epoch": epoch,
                        "val_rel": rel.tolist(),
                        "val_rel_wall": rel_wall.tolist(),
                    },
                    OUT / "best.pt",
                )

    with open(OUT / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"best mean val rel L2: {best:.3f}")
