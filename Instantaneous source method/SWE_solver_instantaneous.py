# Import libraries
import numpy as np
from matplotlib import pyplot as plt


# 2D Shallow Water Equation code for Instantaneous Source
# ------------------------------
def SWE_solver_instantaneous(eta0, qx_init, qy_init, h, g, T_start, T_end, dx, dy, dt, case, plot_min, plot_max):
    """
    Solving 2D Shallow water equation using Instantaneous Source Method .

    Parameters
    ----------
    eta0 : np.ndarray
        The initial wave height field as a 2D array of floats.
    qx_init : np.ndarray
        The initial discharge flux field in x-direction as a 2D array of floats.
    qy_init : np.ndarray
        The initial discharge flux field in y-direction as a 2D array of floats.
    h : np.ndarray
        Bathymetric model as a 2D array of floats.
    g : float
        gravity acceleration.
    T_start : float
        Simulation Start time
    T_end : float
        Simulation end time
    dx : float
        Spatial grid-point distance in x-direction.
    dy : float
        Spatial grid-point distance in y-direction.
    dt : float
        Time step.
    case : basestring
        Name of the case
    plot_min : float
        Limit for plot.
    plot_max : float
        Limit for plot.
    """

    # import
    from update_eta_instantaneous import update_eta_instantaneous
    from update_qx import update_qx
    from update_qy import update_qy

    # Copy fields
    eta = eta0.copy()
    qx = qx_init.copy()
    qy = qy_init.copy()

    # Compute total thickness of water column D
    D = eta + h

    # Estimate number of grid points in x- and y-direction
    ny, nx = eta.shape

    # Time and Result parameter initialization
    t = 0
    output = 0
    result = np.zeros((nx, int(T_end + 1)))

    nt = int((T_end - T_start) / dt)

    for n in range(nt):

        # Update Eta field
        eta = update_eta_instantaneous(eta, qx, qy, dx, dy, dt, nx, ny)

        # Update qx field
        qx = update_qx(eta, qx, D, g, dx, dt, nx, ny)

        # Update qy field
        qy = update_qy(eta, qy, D, g, dy, dt, nx, ny)

        # Compute total water column D
        D = eta + h

        # Updating time and results
        t = t + dt
        output = output + 1

        if output % (1 / dt) == 0:
            result[:, round(T_start + t)] = eta[:, int(ny / 2)]

    # Plotting
    Lx = dx * (nx - 1)
    distance = np.linspace(-Lx / 2000, Lx / 2000, num=nx)
    time = np.linspace(0, T_end, num=int(T_end + 1))
    X, T = np.meshgrid(distance, time)

    f = plt.figure()
    f.set_figwidth(6)
    f.set_figheight(10)
    plt.rcParams.update({'font.size': 14})
    plt.rcParams["font.weight"] = "bold"

    plt.pcolor(X, T, result.T, cmap='coolwarm', shading='auto')
    # plt.pcolor(X, T, result.T, cmap='jet', shading='auto')
    plt.xlabel('Distance (km)', fontweight='bold')
    plt.ylabel('Time (s)', fontweight='bold')
    plt.clim(plot_min, plot_max)
    plt.colorbar()
    plt.savefig("../Output/Instantaneous/Result_" + case, dpi=125)
    # plt.show()

    return None
