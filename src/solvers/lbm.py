import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

jax.config.update("jax_default_matmul_precision", "highest")
# D2Q9 lattice: directions (cx, cy) and weights
C_NP = np.array(
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
)

C = jnp.array(C_NP, dtype=jnp.float32)
W = jnp.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4, dtype=jnp.float32)
OPP = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

# opposite direction of each i, for bounce-back
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

CS2 = 1 / 3  # lattice speed of sound squared


def macroscopic(f):
    rho = f.sum(axis=0)
    momentum = jnp.einsum("id,iyx->dyx", C, f)
    u = momentum / rho
    return rho, u


def equilibrium(rho, u):
    cu = jnp.einsum("id,dyx->iyx", C, u)
    usq = (u**2).sum(0)
    return W[:, None, None] * rho * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * usq)


def collide(f, feq, tau):
    return f - (f - feq) / tau


def stream(f):
    return jnp.stack(
        [jnp.roll(f[i], shift=(cy, cx), axis=(0, 1)) for i, (cx, cy) in enumerate(C_NP)]
    )


def cylinder_mask(Ny, Nx, cx, cy, r):
    y, x = jnp.mgrid[0:Ny, 0:Nx]
    return (x - cx) ** 2 + (y - cy) ** 2 <= r**2


def step(f, tau, mask, u_in):
    f = stream(f)
    f = f.at[:, :, -1].set(f[:, :, -2])  # outlet
    bounced = f[OPP]  # reversed populations

    rho, u = macroscopic(f)
    f = f - (f - equilibrium(rho, u)) / tau  # collide

    Ny = f.shape[1]
    u_inlet = jnp.zeros((2, Ny, 1)).at[0].set(u_in)
    f = f.at[:, :, :1].set(equilibrium(jnp.ones((Ny, 1)), u_inlet))  # inlet

    return jnp.where(mask[None], bounced, f)


def run_chunk(f, tau, mask, u_in, n):
    return lax.fori_loop(0, n, lambda _, f: step(f, tau, mask, u_in), f)


run_chunk_batched = jax.jit(
    jax.vmap(run_chunk, in_axes=(0, None, 0, None, None)), static_argnums=4
)
macro_batched = jax.jit(jax.vmap(macroscopic))
equil_batched = jax.jit(jax.vmap(equilibrium))


def solve_batch(
    masks, D, Re, u_in=0.1, max_steps=60_000, chunk=2_000, tol=4e-6, kick=False
):
    """masks: NumPy bool (B, Ny, Nx). Returns NumPy rho (B, Ny, Nx), u (B, 2, Ny, Nx),
    converged (B,), steps taken."""
    B, Ny, Nx = masks.shape
    tau = 3 * (u_in * D / Re) + 0.5
    m = jnp.asarray(masks)

    rho = jnp.ones((B, Ny, Nx), dtype=jnp.float32)
    u = jnp.zeros((B, 2, Ny, Nx), dtype=jnp.float32).at[:, 0].set(u_in)
    if kick:  # only needed to trigger shedding in the validation run
        u = u.at[:, 1].add(1e-3 * jnp.sin(2 * jnp.pi * jnp.arange(Nx) / Nx))
    f = equil_batched(rho, u)

    _, u_old = macro_batched(f)
    steps = 0
    while steps < max_steps:
        f = run_chunk_batched(f, tau, m, u_in, chunk)
        steps += chunk
        rho, u = macro_batched(f)
        u = jnp.where(m[:, None], 0.0, u)
        change = np.asarray(
            jnp.abs(u - u_old).max(axis=(1, 2, 3))
        )  # one value per shape
        if (change < tol).all():
            break
        u_old = u

    return np.asarray(rho), np.asarray(u), change < tol, steps
