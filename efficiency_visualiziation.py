# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 15:11:59 2026

@author: david
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Motor torque vs. efficiency 
effcy_tau = np.array([0, 10, 20, 40, 70, 165])
effcy = np.array([0, 0.55, 0.75, 0.71, 0.50, 0.05])

# Cubic interpolation function
effcy_fun = interp1d(effcy_tau, effcy, kind='cubic')

# Generate 100 evenly spaced torque values
torque_values = np.linspace(
    np.min(effcy_tau),
    np.max(effcy_tau),
    100
)

# Interpolated efficiency values
efficiency_values = effcy_fun(torque_values)

# Plot original data using star symbols
plt.plot(effcy_tau, effcy, '*', label='Original Data')

# Plot interpolated efficiency curve
plt.plot(torque_values, efficiency_values,
         '-', label='Cubic Interpolation')

# Label axes
plt.xlabel('Motor Torque (N·m)')
plt.ylabel('Motor Efficiency')

# Add title, legend, and grid
plt.title('Motor Efficiency vs. Torque')
plt.legend()
plt.grid(True)

# Display graph
plt.show()
