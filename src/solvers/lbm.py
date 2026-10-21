import numpy as np

# D2Q9 lattice: directions (cx, cy) and weights
C = np.array(
    [
        [0, 0],  # 0: rest
        [1, 0],  # 1: east
        [0, 1],  # 2: north
        [-1, 0],  # 3: west
        [0, -1],  # 4: south
        [1, 1],  # 5: north-east
        [-1, 1],  # 6: north-west
        [-1, -1],  # 7: south-west
        [1, -1],
    ]
)  # 8: south-east

W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])

# opposite direction of each i, for bounce-back
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

CS2 = 1 / 3  # lattice speed of sound squared


def macroscopic(f):
    rho = f.sum(axis=0)
    momentum = np.einsum("id,iyx->dyx", C, f)
    u = momentum / rho
    return rho, u


def equilibrium(rho, u):
    cu = np.einsum("id,dyx->iyx", C, u)
    usq = (u**2).sum(0)
    return W[:, None, None] * rho * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * usq)


def collide(f, feq, tau):
    return f - (f - feq) / tau


def stream(f):
    out = np.empty_like(f)
    for i, (cx, cy) in enumerate(C):
        out[i] = np.roll(f[i], shift=(cy, cx), axis=(0, 1))
    return out


def cylinder_mask(Ny, Nx, cx, cy, r):
    y, x = np.mgrid[0:Ny, 0:Nx]
    return (x - cx) ** 2 + (y - cy) ** 2 <= r**2


def step(f, tau, mask, u_in):
    f = stream(f)

    f[:, :, -1] = f[:, :, -2]
    bounced = f[OPP][:, mask]

    rho, u = macroscopic(f)
    feq = equilibrium(rho, u)
    f = collide(f, feq, tau)

    Ny = f.shape[1]
    rho_in = np.ones((Ny, 1))
    u_inlet = np.zeros((2, Ny, 1))
    u_inlet[0] = u_in
    f[:, :, :1] = equilibrium(rho_in, u_inlet)

    f[:, mask] = bounced
    return f


def run(
    Nx=400, Ny=100, D=20, Re=20, u_in=0.1, n_steps=20_000, check_every=500, tol=1e-6
):
    nu = u_in * D / Re
    tau = 3 * nu + 0.5
    mask = cylinder_mask(Ny, Nx, cx=Nx // 4, cy=Ny // 2, r=D / 2)

    rho = np.ones((Ny, Nx))
    u = np.zeros((2, Ny, Nx))
    u[0] = u_in
    u[1] += 1e-3 * np.sin(
        2 * np.pi * np.arange(Nx) / Nx
    )  # small kick to trigger shedding at high Re
    f = equilibrium(rho, u)

    u_old = u.copy()
    for t in range(1, n_steps + 1):
        f = step(f, tau, mask, u_in)
        if t % check_every == 0:
            rho, u = macroscopic(f)
            u[:, mask] = 0
            change = np.abs(u - u_old).max()
            print(f"step {t}: max change {change:.2e}")
            if change < tol:
                break
            u_old = u.copy()

    rho, u = macroscopic(f)
    u[:, mask] = 0
    return rho, u, mask
