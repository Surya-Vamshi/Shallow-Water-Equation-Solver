# Import libraries
from numba import jit


# Update qy flux
@jit(nopython=True)
def update_qy(eta, qy, D, g, dy, dt, nx, ny):
    # loop over spatial grid
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            detady = (eta[j + 1, i] - eta[j - 1, i]) / (2. * dy)

            # update N field
            qy[j, i] = qy[j, i] - dt * (g * D[j, i] * detady)

    # apply Neumann boundary conditions for qx at all boundaries
    qy[0, :] = qy[1, :]
    qy[-1, :] = qy[-2, :]
    qy[:, 0] = qy[:, 1]
    qy[:, -1] = qy[:, -2]

    return qy
