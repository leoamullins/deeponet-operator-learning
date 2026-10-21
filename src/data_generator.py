import scipy


def compute_antiderivative(u, x_fine):
    return scipy.integrate.cumulative_trapezoid(u, x_fine, axis=1, initial=0)
