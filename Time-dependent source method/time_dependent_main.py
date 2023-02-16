# Import libraries
import numpy as np
import math

from SWE_solver_time_dependent import SWE_solver_time_dependent


"""====================== Time Dependent Source Method ====================================="""
# ======================= Start of Common parameter to change ==============================
# Number of Grid-points in each direction
nx = 401  # number of points in the x direction
ny = 401  # number of points in the y direction

# Time-step for the simulation
dt = 0.1

# define some constants
g = 9.81  # gravity acceleration [m/s^2]

# ======================= End of Common parameter to change =================================

"""====================== Scenario 1 ================================================"""
# ======================= Start of parameter to change ==============================
# Definition of Time-dependent source method parameters
Lx = 200000.0  # Domain size in the x direction [m]
Ly = 200000.0  # Domain size in the y direction [m]

# characterized spatial width and duration of source
sigma_r = 12500  # (m)
sigma_t = 125  # (sec)

# Maximum wave propagation time [s]
T_end = 800.

# Case name and axis limit
case = "Scenario_1"
plot_min = -0.2
plot_max = 0.2

# ======================= End of parameter to change =================================

dx = Lx / (nx - 1)  # grid spacing in the x direction []
dy = Ly / (ny - 1)  # grid spacing in the y direction []

# Define the locations along a gridline.
x = np.linspace(-Lx / 2, Lx / 2, num=nx)
y = np.linspace(-Ly / 2, Ly / 2, num=ny)

# Define initial eta, qx, qy
X, Y = np.meshgrid(x, y)  # coordinates X,Y required to define eta, h, qx, qy

# Define source term
dbdt_grid = (1 / (sigma_t * math.sqrt(2 * math.pi))) * np.exp(-((X ** 2) + (Y ** 2)) / (2 * sigma_r * sigma_r))

# Define constant ocean depth profile h = 4000 m
h = 4000.0 * np.ones_like(X)

# Define initial eta [m]
eta0 = 0. * np.exp(-(X ** 2) - (Y ** 2))

# Define initial fluxes
qx0 = 0. * eta0
qy0 = 0. * eta0

# Running Simulation Case
SWE_solver_time_dependent(eta0, qx0, qy0, h, g, T_end, dx, dy, dt, dbdt_grid, sigma_t, case, plot_min, plot_max)

print("Completed Time Dependent Scenario 1 simulation")

"""====================== Scenario 2 ================================================"""
# ======================= Start of parameter to change ==============================
# Definition of Time-dependent source method parameters
Lx = 200000.0  # Domain size in the x direction [m]
Ly = 200000.0  # Domain size in the y direction [m]

# characterized spatial width and duration of source
sigma_r = 12500  # (m)
sigma_t = 1.25  # (sec)

# Maximum wave propagation time [s]
T_end = 400.

# Case name and axis limit
case = "Scenario_2"
plot_min = -1
plot_max = 1

# ======================= End of parameter to change =================================

dx = Lx / (nx - 1)  # grid spacing in the x direction []
dy = Ly / (ny - 1)  # grid spacing in the y direction []

# Define the locations along a gridline.
x = np.linspace(-Lx / 2, Lx / 2, num=nx)
y = np.linspace(-Ly / 2, Ly / 2, num=ny)

# Define initial eta, qx, qy
X, Y = np.meshgrid(x, y)  # coordinates X,Y required to define eta, h, qx, qy

# Define source term
dbdt_grid = (1 / (sigma_t * math.sqrt(2 * math.pi))) * np.exp(-((X ** 2) + (Y ** 2)) / (2 * sigma_r * sigma_r))

# Define constant ocean depth profile h = 4000 m
h = 4000.0 * np.ones_like(X)

# Define initial eta [m]
eta0 = 0. * np.exp(-(X ** 2) - (Y ** 2))

# Define initial fluxes
qx0 = 0. * eta0
qy0 = 0. * eta0


# Running Simulation Case
SWE_solver_time_dependent(eta0, qx0, qy0, h, g, T_end, dx, dy, dt, dbdt_grid, sigma_t, case, plot_min, plot_max)

print("Completed Time Dependent Scenario 2 simulation")

"""====================== Scenario 3 ================================================"""
# ======================= Start of parameter to change ==============================
# Definition of Time-dependent source method parameters
Lx = 40000.0  # Domain size in the x direction [m]
Ly = 40000.0  # Domain size in the y direction [m]

# characterized spatial width and duration of source
sigma_r = 1250  # (m)
sigma_t = 1.25  # (sec)

# Maximum wave propagation time [s]
T_end = 150.

# Case name and axis limit
case = "Scenario_3"
plot_min = -0.25
plot_max = 0.25

# ======================= End of parameter to change =================================

dx = Lx / (nx - 1)  # grid spacing in the x direction []
dy = Ly / (ny - 1)  # grid spacing in the y direction []

# Define the locations along a gridline.
x = np.linspace(-Lx / 2, Lx / 2, num=nx)
y = np.linspace(-Ly / 2, Ly / 2, num=ny)

# Define initial eta, qx, qy
X, Y = np.meshgrid(x, y)  # coordinates X,Y required to define eta, h, qx, qy

# Define source term
dbdt_grid = (1 / (sigma_t * math.sqrt(2 * math.pi))) * np.exp(-((X ** 2) + (Y ** 2)) / (2 * sigma_r * sigma_r))

# Define constant ocean depth profile h = 4000 m
h = 4000.0 * np.ones_like(X)

# Define initial eta [m]
eta0 = 0. * np.exp(-(X ** 2) - (Y ** 2))

# Define initial fluxes
qx0 = 0. * eta0
qy0 = 0. * eta0

# Running Simulation Case
SWE_solver_time_dependent(eta0, qx0, qy0, h, g, T_end, dx, dy, dt, dbdt_grid, sigma_t, case, plot_min, plot_max)

print("Completed Time Dependent Scenario 3 simulation")
