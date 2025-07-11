# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:41:21 2024

@author: KHChan
"""

from scipy.optimize import minimize
import numpy as np

# Try relative import first
try:
    from .distance import compute_angular_distance
except ImportError:
    from distance import compute_angular_distance

def objective_function(new_points, given_points, target_distances):
    new_points_reshaped = new_points.reshape(2, 2)
    angles_matrix = compute_angular_distance(new_points_reshaped,given_points)
    return np.sum((angles_matrix - target_distances) ** 2)

def optimize_laue_points(initial_guess, given_points, target_distances):
    # Optimize to find the new points
    bounds = [(0, 180), (-180, 180)] * 2  # Bounds for two points
    result = minimize(objective_function, initial_guess, args=(given_points, target_distances), bounds=bounds)
    new_points = result.x.reshape(2, 2)
    return new_points, objective_function(new_points, given_points, target_distances)

