# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 12:11:18 2026

@author: jwkno
"""
from scipy.special import erf
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# from phase 1
def _require_dict(x, name):
    if not isinstance(x, dict):
        raise Exception('%s must be a dict.' % name)
        
def get_gear_ratio(speed_reducer):
    """Gear ratio Ng of a reverted gear train: (d2/d1)^2."""
    _require_dict(speed_reducer, 'speed_reducer')
    if str(speed_reducer['type']).strip().lower() != 'reverted':
        raise Exception("Unsupported speed reducer type '%s'. Only 'reverted' is "
                        "valid in this phase." % speed_reducer['type'])
    return (speed_reducer['diam_gear'] / speed_reducer['diam_pinion']) ** 2



def motorW(v,rover):
    if type(rover) != dict:
        raise Exception('Second input must be a dictionary')
        
    if type(v) == np.ndarray:
        # make sure the array is 1D (not a matrix)
        if len(v.shape) != 1:
            raise Exception('v cannot be a matrix, it must be a 1D array')

    # if v is not an array, check that it is a single number
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

   
    
    