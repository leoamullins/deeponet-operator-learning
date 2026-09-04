import torch
import torch.nn as nn


class DeepONet(nn.Module):
    def __init__(self, m, p=40, width=40):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(m, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, p),
        )
        self.trunk = nn.Sequential(
            nn.Linear(1, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, p),
            nn.Tanh(),
        )
        self.b0 = nn.Parameter(torch.zeros(1))

    def forward(self, u_sensors, y_query):
        b = self.branch(u_sensors)
        t = self.trunk(y_query)
        raw = b @ t.T + self.b0

        t0 = self.trunk(torch.zeros(1, 1, dtype=y_query.dtype))
        pred0 = b @ t0.T + self.b0

        return raw - pred0
