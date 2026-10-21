import torch
from torch import nn


class DeepONet(nn.Module):
    def __init__(self, m, p=40, n_out=3, width=40, coord_dim=2, zero_at_origin=False):
        super().__init__()
        self.p = p
        self.n_out = n_out
        self.zero_at_origin = zero_at_origin
        self.coord_dim = coord_dim
        # (n_samples, coord_dim) -> (n_samples, width) -> ... -> (final) (n_samples, self.n_out * self.p)
        self.branch = nn.Sequential(
            nn.Linear(m, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, self.n_out * self.p),
        )
        self.trunk = nn.Sequential(
            nn.Linear(coord_dim, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, self.n_out * self.p),
            nn.Tanh(),
        )
        self.b0 = nn.Parameter(torch.zeros(self.n_out))

    def forward(self, sdf, xy):
        b = self.branch(sdf)
        t = self.trunk(xy)

        # b: (n_samples, n_out * p) -> (n_samples, n_out, p)
        # t: (n_points, n_out * p)  -> (n_points, n_out, p)
        b = b.view(-1, self.n_out, self.p)
        t = t.view(-1, self.n_out, self.p)
        out = torch.einsum("sfk,nfk->snf", b, t)

        # allow for fixing the zero to origin.
        if self.zero_at_origin:
            y0 = torch.zeros(1, self.coord_dim, dtype=xy.dtype, device=xy.device)
            t0 = self.trunk(y0).view(-1, self.n_out, self.p)
            pred0 = torch.einsum("sfk,nfk->snf", b, t0)
            out = out - pred0

        return torch.squeeze(out + self.b0, -1)
