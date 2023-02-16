# Import libraries
from numba import jit


# Update qx flux
@jit(nopython=True)
def update_qx(eta, qx, D, g, dx, dt, nx, ny):
    # loop over spatial grid
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            # compute spatial derivatives
            detadx = (eta[j, i + 1] - eta[j, i - 1]) / (2. * dx)

            # update M field
            qx[j, i] = qx[j, i] - dt * (g * D[j, i] * detadx)

    # apply Neumann boundary conditions for qx at all boundaries
    qx[0, :] = qx[1, :]
    qx[-1, :] = qx[-2, :]
    qx[:, 0] = qx[:, 1]
    qx[:, -1] = qx[:, -2]

    return qx
