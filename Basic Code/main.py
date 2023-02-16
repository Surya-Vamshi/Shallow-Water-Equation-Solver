# Import libraries
import numpy as np
import math
from matplotlib import pyplot
from scipy.ndimage import gaussian_filter

# import JIT from Numba
from numba import jit

# import
from Shallow_water_solver import Shallow_water_solver

# Set the font family and size to use for Matplotlib figures.
pyplot.rcParams['font.family'] = 'serif'
pyplot.rcParams['font.size'] = 16


# Definition of modelling parameters
# ----------------------------------
Lx = 200000.0   # width of the mantle in the x direction []
Ly = 200000.0   # thickness of the mantle in the y direction []
nx = 401     # number of points in the x direction
ny = 401     # number of points in the y direction
dx = Lx / (nx - 1)  # grid spacing in the x direction []
dy = Ly / (ny - 1)  # grid spacing in the y direction []

# Define the locations along a gridline.
x = np.linspace(0.0, Lx, num=nx)
y = np.linspace(0.0, Ly, num=ny)

# Define initial eta, M, N
X, Y = np.meshgrid(x, y)  # coordinates X,Y required to define eta, h, M, N

# characterize spatial width and duration of source
sigma_r = 12500.0  # (m)
sigma_t = 1.25  # (sec)

# Define source term

# Define constant ocean depth profile h = 4000 m
h = 4000.0 * np.ones_like(X)
# h = 50 - 45 * numpy.tanh((X-70.)/8.)

# Define initial eta [m]
eta0 = 1000 * np.exp(-((X-100000)**2/10000)-((Y-100000)**2/20000))

# Define initial M and N
M0 = 1000. * eta0
N0 = 0. * M0


# define some constants
g = 9.81  # gravity acceleration [m/s^2]

# Maximum wave propagation time [s]
Tmax = 400.
dt = 1.
cfl = dx / (2*g*np.max(h))**0.5
nt = (int)(Tmax/dt)


# Compute eta, qx, qy fields
eta, M, N = Shallow_water_solver(eta0, M0, N0, h, g, nt, dx, dy, dt, X, Y)

# print("Done: ", len(eta_list))