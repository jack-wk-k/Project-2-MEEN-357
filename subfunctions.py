# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 12:11:18 2026

@author: jwkno
"""
from scipy.special import erf
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# from phase 1
# (input validation)

def _is_scalar(x):
    return (isinstance(x, (int, float, np.integer, np.floating))
            and not isinstance(x, (bool, np.bool_)))

def _is_scalar_or_vector(x):
    if _is_scalar(x):
        return True
    if isinstance(x, np.ndarray):
        return x.ndim <= 1 and np.issubdtype(x.dtype, np.number)
    return False

def _finish(result, original):
    """Return a float if the original input was a scalar, else an ndarray."""
    if _is_scalar(original) or (isinstance(original, np.ndarray) and original.ndim == 0):
        return float(result)
    return np.asarray(result, dtype=float)

def _require_dict(x, name):
    if not isinstance(x, dict):
        raise Exception('%s must be a dict.' % name)

def _require_angles(angle):
    if not _is_scalar_or_vector(angle):
        raise Exception('terrain_angle must be a scalar or a 1D numpy array.')
    if np.any(np.abs(np.asarray(angle, dtype=float)) > 75):
        raise Exception('All terrain angles must be between -75 and +75 degrees.')

def _require_same_size(a, b):
    if np.shape(a) != np.shape(b):
        raise Exception('omega and terrain_angle must be the same size.')

def _require_Crr(Crr):
    if not _is_scalar(Crr) or Crr <= 0:
        raise Exception('Crr must be a positive scalar.')

# Required functions
def tau_dcmotor(omega, motor):
    """Motor shaft torque [Nm] for motor shaft speed omega [rad/s]."""
    if not _is_scalar_or_vector(omega):
        raise Exception('omega must be a scalar or a 1D numpy array.')
    _require_dict(motor, 'motor')

    w = np.asarray(omega, dtype=float)
    tau_s = motor['torque_stall']
    tau_nl = motor['torque_noload']
    w_nl = motor['speed_noload']

    tau = tau_s - ((tau_s - tau_nl) / w_nl) * w
    tau = np.where(w < 0, tau_s, tau)      # spinning backwards -> stall torque
    tau = np.where(w > w_nl, 0.0, tau)     # faster than no-load -> no torque
    return _finish(tau, omega)

def get_gear_ratio(speed_reducer):
    """Gear ratio Ng of a reverted gear train: (d2/d1)^2."""
    _require_dict(speed_reducer, 'speed_reducer')
    if str(speed_reducer['type']).strip().lower() != 'reverted':
        raise Exception("Unsupported speed reducer type '%s'. Only 'reverted' is "
                        "valid in this phase." % speed_reducer['type'])
    return (speed_reducer['diam_gear'] / speed_reducer['diam_pinion']) ** 2

def get_mass(rover):
    """Total rover mass [kg] (chassis, power, payload, 6 wheel assemblies)."""
    _require_dict(rover, 'rover')
    wa = rover['wheel_assembly']
    per_wheel = wa['wheel']['mass'] + wa['speed_reducer']['mass'] + wa['motor']['mass']
    return (rover['chassis']['mass'] + rover['power_subsys']['mass']
            + rover['science_payload']['mass'] + 6 * per_wheel)

def F_drive(omega, rover):
    """Combined drive force [N] from all six wheels for motor shaft speed omega."""
    if not _is_scalar_or_vector(omega):
        raise Exception('omega must be a scalar or a 1D numpy array.')
    _require_dict(rover, 'rover')

    wa = rover['wheel_assembly']
    tau_in = tau_dcmotor(omega, wa['motor'])
    Ng = get_gear_ratio(wa['speed_reducer'])
    tau_out = Ng * tau_in
    Fd = 6 * tau_out / wa['wheel']['radius']
    return _finish(Fd, omega)

def F_gravity(terrain_angle, rover, planet):
    """Gravity force [N] along the direction of travel (uphill -> negative)."""
    _require_angles(terrain_angle)
    _require_dict(rover, 'rover')
    _require_dict(planet, 'planet')

    ang = np.asarray(terrain_angle, dtype=float)
    Fgt = -get_mass(rover) * planet['g'] * np.sin(np.radians(ang))
    return _finish(Fgt, terrain_angle)

def F_rolling(omega, terrain_angle, rover, planet, Crr):
    """Rolling resistance force [N] summed over six wheels (always <= 0)."""
    if not _is_scalar_or_vector(omega) or not _is_scalar_or_vector(terrain_angle):
        raise Exception('omega and terrain_angle must be scalars or 1D numpy arrays.')
    _require_same_size(omega, terrain_angle)
    _require_angles(terrain_angle)
    _require_dict(rover, 'rover')
    _require_dict(planet, 'planet')
    _require_Crr(Crr)

    wa = rover['wheel_assembly']
    Ng = get_gear_ratio(wa['speed_reducer'])
    v = wa['wheel']['radius'] * np.asarray(omega, dtype=float) / Ng   # rover speed
    ang = np.asarray(terrain_angle, dtype=float)
    Fn = get_mass(rover) * planet['g'] * np.cos(np.radians(ang))       # total normal force
    Frr = -erf(40 * v) * Crr * Fn                                      # 6 x (Fn/6 * Crr)
    return _finish(Frr, omega)

def F_net(omega, terrain_angle, rover, planet, Crr):
    """Net force [N] on the rover in its direction of motion."""
    if not _is_scalar_or_vector(omega) or not _is_scalar_or_vector(terrain_angle):
        raise Exception('omega and terrain_angle must be scalars or 1D numpy arrays.')
    _require_same_size(omega, terrain_angle)
    _require_angles(terrain_angle)
    _require_dict(rover, 'rover')
    _require_dict(planet, 'planet')
    _require_Crr(Crr)

    Fnet = (F_drive(omega, rover)
            + F_gravity(terrain_angle, rover, planet)
            + F_rolling(omega, terrain_angle, rover, planet, Crr))
    return _finish(Fnet, omega)




def motorW(v,rover):
    if type(rover) != dict:
        raise Exception('Second input must be a dictionary')
        
    if type(v) == np.ndarray:
        #make sure the array is 1D (not a matrix)
        if len(v.shape) != 1:
            raise Exception('v cannot be a matrix, it must be a 1D array')

    #if v is not an array, check that it is a single number
    else:
        if type(v) == int:
            pass  # int is okay
        elif type(v) == float:
            pass  # float is okay
        else:
            raise Exception('First input must be a scalar or a numpy array')
    
    # define wheel ratio and and radius
    r = rover['wheel_assembly']['wheel']['radius']
    Ng = get_gear_ratio(rover['wheel_assembly']['speed_reducer'])
    
    # calculate based on if array or scalar
    if type(v) == np.ndarray:
        how_many_items = len(v) #check if array
        new_list = [] 
        for i in range(how_many_items):
            w_i = Ng * v[i] / r
            new_list.append(w_i)
        return np.array(new_list)
    else: #if not array treat as scalar
        w = Ng * v / r
        return w

def rover_dynamics(t, y, rover, planet, experiment):
    v = y[0]   #velocity
    r = y[1]   #position
    
    #fit the graph for the terrain and find the angle at the rover's position
    alpha_fun = interp1d(experiment['alpha_dist'], experiment['alpha_deg'],
                         kind='cubic', fill_value='extrapolate')
    
    terrain_angle = float(alpha_fun(r))

    #motor speed from the rover velocity
    omega = motorW(v, rover) 

    #net force and mass of rover
    Fnet = F_net(omega, terrain_angle, rover, planet, experiment['Crr'])
    m= get_mass(rover)

    #Newton's second law
    accel = Fnet / m
  

    
    dydt = np.array([accel,v])
    return dydt
    