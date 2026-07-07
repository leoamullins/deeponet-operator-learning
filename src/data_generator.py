import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


def generate_data(n_samples, m_sensors, n_queries):
    kernel = RBF(length_scale=0.2)
    gp = GaussianProcessRegressor(kernel=kernel)

    sensors = np.linspace(0, 1, m_sensors)

    u_data = []  # branch inputs
    x_data = []  # trunk inputs
    s_data = []  # targets

    for _ in range(n_samples):
        # sample random input function
        u = gp.sample_y(sensors.reshape(-1, 1), n_samples=1).flatten()

        # random point
        x_query = np.random.uniform(0, 1)

        # compute target int from 0 to x_query
        mask = sensors <= x_query
        s = np.trapezoid(u[mask], sensors[mask])

        u_data.append(u)
        x_data.append([x_query])
        s_data.append([s])

    return u_data, x_data, s_data
