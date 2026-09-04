import numpy as np
from scipy.linalg import cholesky
import scipy


def build_cholesky_factor(x_fine, ell, jitter=1e-8):
    n_fine = x_fine.shape[0]

    # use broadcasting to make pairwise distance matrix
    D2 = (x_fine[:, None] - x_fine[None, :]) ** 2

    # calculate the kernel
    K = np.exp(-D2 / (2 * ell**2))

    # adding the jitter to avoid, lin alg error
    K += jitter * np.eye(n_fine)
    return cholesky(K, lower=True)


def sample_functions(L, n_functions, rng):
    n = L.shape[0]
    z = rng.standard_normal((n, n_functions))

    # transform z with cholesky factor L
    u = L @ z
    return u.T


def compute_antiderivative(u, x_fine):
    return scipy.integrate.cumulative_trapezoid(u, x_fine, axis=1, initial=0)
