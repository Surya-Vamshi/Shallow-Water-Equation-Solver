# import JIT from Numba
from numba import jit

# Update Eta field
@jit(nopython=True)
def update_eta_instantaneous(eta, qx, qy, dx, dy, dt, nx, ny):
    # loop over spatial grid
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            # compute spatial derivatives
            dqxdx = (qx[j, i + 1] - qx[j, i - 1]) / (2. * dx)
            dqydy = (qy[j + 1, i] - qy[j - 1, i]) / (2. * dy)

            # update eta field
            eta[j, i] = eta[j, i] - dt * (dqxdx + dqydy)

    # apply Neumann boundary conditions for eta at all boundaries
    eta[0, :] = eta[1, :]
    eta[-1, :] = eta[-2, :]
    eta[:, 0] = eta[:, 1]
    eta[:, -1] = eta[:, -2]

    return eta
