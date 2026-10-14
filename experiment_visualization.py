
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from define_experiment import experiment1

experiment, end_event = experiment1()

alpha_dist = experiment['alpha_dist']   #positions
alpha_deg = experiment['alpha_deg']     #terrain angles

alpha_fun = interp1d(alpha_dist, alpha_deg, kind= 'cubic', fill_value ='extrapolate') #fit the cubic spline

x_new = np.linspace(np.min(alpha_dist), np.max(alpha_dist), 100)


y_new = alpha_fun(x_new) #find the terrain angle at each of those positions

fig, ax = plt.subplots()
ax.plot(alpha_dist, alpha_deg, '*', label='Data points')
ax.plot(x_new, y_new, '-', label='Cubic spline')
ax.set_title('Terrain angle vs. position')
ax.set_xlabel('Position (m)')
ax.set_ylabel('Terrain angle (deg)')
ax.legend()
