# import JIT from Numba
from numba import jit
import math
import numpy as np

# Update Eta field
@jit(nopython=True)
def update_eta(eta, qx, qy, dx, dy, dt, nx, ny):
    A = 200000 * 200000
    sigma_r = 12500.0  # (m)
    sigma_t = 1.25  # (sec)

    # loop over spatial grid
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            # compute spatial derivatives
            dqxdx = (qx[j, i + 1] - qx[j, i - 1]) / (2. * dx)
            dqydy = (qy[j + 1, i] - qy[j - 1, i]) / (2. * dy)


            dbdt = (A * sigma_t * math.sqrt(2 * math.pi))

            # update eta field
            eta[j, i] = eta[j, i] - dt * (dqxdx + dqydy)

    # apply Neumann boundary conditions for eta at all boundaries
    eta[0, :] = eta[1, :]
    eta[-1, :] = eta[-2, :]
    eta[:, 0] = eta[:, 1]
    eta[:, -1] = eta[:, -2]

    return eta
