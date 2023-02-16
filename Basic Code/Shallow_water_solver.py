# Import libraries
import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import animation
from scipy.ndimage import gaussian_filter
#%matplotlib inline

# import JIT from Numba
from numba import jit

eta_list = list()
# 2D Shallow Water Equation code with JIT optimization
# -------------------------------------------------------
def Shallow_water_solver(eta0, qx_init, qy_init, h, g, nt, dx, dy, dt, X, Y):
    """
    Computes and returns the discharge fluxes M, N and wave height eta from
    the 2D Shallow water equation using the FTCS finite difference method.

    Parameters
    ----------
    eta0 : np.ndarray
        The initial wave height field as a 2D array of floats.
    qx_init : np.ndarray
        The initial discharge flux field in x-direction as a 2D array of floats.
    qy_init : np.ndarray
        The initial discharge flux field in y-direction as a 2D array of floats.
    h : np.ndarray
        Bathymetry model as a 2D array of floats.
    g : float
        gravity acceleration.
    alpha : float
        Manning's roughness coefficient.
    nt : integer
        Number fo timesteps.
    dx : float
        Spatial gridpoint distance in x-direction.
    dy : float
        Spatial gridpoint distance in y-direction.
    dt : float
        Time step.
    X : np.ndarray
        x-coordinates as a 2D array of floats.
    Y : np.ndarray
        y-coordinates as a 2D array of floats.

    Returns
    -------
    eta : np.ndarray
        The final wave height field as a 2D array of floats.
    qx : np.ndarray
        The final discharge flux field in x-direction as a 2D array of floats.
    qy : np.ndarray
        The final discharge flux field in y-direction as a 2D array of floats.
    """

    # import
    from update_eta import update_eta
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

    # Define the locations along a grid line.
    x = np.linspace(0, nx * dx, num=nx)
    y = np.linspace(0, ny * dy, num=ny)

    anim_interval = 20
    snap_count = 0

    # Printing the initial case
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X, Y, eta, cmap='winter', rstride=1, cstride=1, alpha=None, antialiased=True)
    fig.colorbar(surf)
    ax.plot_surface(X, Y, -h, cmap='autumn', rstride=1, cstride=1, alpha=None, antialiased=True)

    # Save plot
    name_snap = "output/Tsunami_" + "%0.*f" % (0, np.fix(snap_count + 1000)) + ".tiff"
    plt.savefig(name_snap, format='tiff', bbox_inches='tight', dpi=125)
    snap_count += 1
    plt.close()

    t = 0

    for n in range(nt):

        # Update Eta field
        eta = update_eta(eta, qx, qy, dx, dy, dt, nx, ny)

        # Update  field
        qx = update_qx(eta, qx, D, g, dx, dt, nx, ny)

        # Update N field
        qy = update_qy(eta, qy, D, g, dy, dt, nx, ny)

        # Compute total water column D
        D = eta + h

        if (n % anim_interval == 0):
            fig = plt.figure(figsize=(10, 10))
            ax = plt.axes(projection='3d')
            surf = ax.plot_surface(X, Y, eta, cmap='winter', rstride=1, cstride=1, alpha=None, antialiased=True)
            fig.colorbar(surf)
            ax.plot_surface(X, Y, -h, cmap='autumn', rstride=1, cstride=1, alpha=None, antialiased=True)
            # Show plot
            # plt.show()

            # Save plot
            name_snap = "output/Tsunami_" + "%0.*f" % (0, np.fix(snap_count + 1000)) + ".tiff"
            plt.savefig(name_snap, format='tiff', bbox_inches='tight', dpi=125)
            snap_count += 1
            plt.close()

    return eta, qx, qy
