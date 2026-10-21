import numpy as np
from scipy.linalg import cholesky


def build_cholesky_factor(x, ell, jitter=1e-8, periodic=False):
    diff = x[:, None] - x[None, :]  # pairwise differences, (n, n)

    if periodic:
        K = np.exp(-2 * np.sin(diff / 2) ** 2 / ell**2)
    else:
        K = np.exp(-(diff**2) / (2 * ell**2))

    K += jitter * np.eye(len(x))  # keeps K positive definite numerically
    return cholesky(K, lower=True)


def sample_functions(L, n_functions, rng):
    z = rng.standard_normal((L.shape[0], n_functions))
    return (L @ z).T
