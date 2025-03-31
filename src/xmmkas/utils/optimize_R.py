# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 12:09:25 2024

@author: tkp20
"""

import numpy as np
from scipy.optimize import leastsq
from scipy.spatial.transform import Rotation as R

# Define constants for degree-radian conversion
DEG = np.pi / 180

def convert_angle_to_direction(twotheta_chi):
    """
    Convert twotheta and chi angles to Cartesian direction vectors.

    Parameters:
        twotheta_chi (ndarray): Array of [twotheta, chi] angles in degrees.

    Returns:
        ndarray: Unit direction vectors corresponding to the input angles.
    """
    twotheta, chi = twotheta_chi.T
    
    theta_rad = (twotheta / 2) * DEG
    chi_rad = chi * DEG
    
    x = -np.sin(theta_rad)
    y = np.cos(theta_rad) * np.sin(chi_rad)
    z = np.cos(theta_rad) * np.cos(chi_rad)
    
    return np.vstack((x, y, z)).T

def calculate_simulated_directions(R_matrix, B_matrix, hkls):
    """
    Calculate the simulated directions for each hkl based on the current R matrix and B matrix.

    Parameters:
        R_matrix (ndarray): Rotation matrix (3x3).
        B_matrix (ndarray): B matrix for crystallographic calculations (3x3).
        hkls (ndarray): Miller indices (N x 3).

    Returns:
        ndarray: Simulated unit direction vectors (N x 3).
    """
    directions = (R_matrix @ B_matrix @ hkls.T).T
    # Normalize each direction to unit vectors
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    return directions

def angular_distance(v1, v2):
    """
    Calculate the angular distance between two unit vectors.

    Parameters:
        v1 (ndarray): Simulated unit vectors (N x 3).
        v2 (ndarray): Measured unit vectors (N x 3).

    Returns:
        ndarray: Angular distances in radians (N x 1).
    """
    dot_product = np.clip(np.sum(v1 * v2, axis=1), -1.0, 1.0)
    return np.arccos(dot_product)

def residuals_orientation(params, measured_directions, B_matrix, hkls):
    """
    Residual function for optimizing orientation only.

    Parameters:
        params (array-like): [alpha, beta, gamma] Euler angles in degrees.
        measured_directions (ndarray): Measured unit direction vectors (N x 3).
        B_matrix (ndarray): Original B matrix (3x3).
        hkls (ndarray): Miller indices (N x 3).

    Returns:
        ndarray: Angular differences in radians (N x 1).
    """
    alpha, beta, gamma = params
    
    # Rotation only, no strain
    rotation = R.from_euler('xyz', [alpha, beta, gamma], degrees=True).as_matrix()
    simulated_directions = calculate_simulated_directions(rotation, B_matrix, hkls)
    angular_diffs = angular_distance(simulated_directions, measured_directions)
    return angular_diffs

def residuals_orientation_and_strain(params, measured_directions, B_matrix, hkls):
    """
    Residual function for optimizing both orientation and deviatoric strain,
    now including intensity weighting.

    Parameters:
        params (array-like): [alpha, beta, gamma, E11, E22, E12, E13, E23]
                             alpha, beta, gamma in degrees
                             Eij: parameters for B-matrix construction
        measured_directions (ndarray): Measured unit direction vectors (N x 3).
        B_matrix (ndarray): Original B matrix (3x3).
        hkls (ndarray): Miller indices (N x 3).
        intensities (ndarray): Intensities corresponding to each reflection (N x 1).

    Returns:
        ndarray: Weighted angular differences in radians (N x 1).
    """
    alpha, beta, gamma, E11, E22, E12, E13, E23 = params

    # Construct rotation matrix from Euler angles
    rotation = R.from_euler('xyz', [alpha, beta, gamma], degrees=True).as_matrix()

    # Construct the refined upper-triangular B-matrix
    B_strained = np.array([[1.0,   E12,  E13],
                           [0.0,   E11,  E23],
                           [0.0,   0.0,  E22]])

    # If you need the original B_matrix as a baseline, you could combine them:
    # B_strained = B_matrix @ B_strained
    # Otherwise, just use B_strained as defined.

    # Compute simulated directions given orientation and new B matrix
    simulated_directions = calculate_simulated_directions(rotation, B_strained, hkls)

    # Compute angular differences between simulated and measured directions
    angular_diffs = angular_distance(simulated_directions, measured_directions)

    # Weighted residuals
    residuals = angular_diffs

    return residuals

def optimize_orientation(measured_twotheta_chi, B_matrix, hkls, initial_euler_angles=(0.0, 0.0, 0.0)):
    """
    Optimize only the orientation (Euler angles) to minimize angular deviations.
    This is used during indexing for more robust orientation determination.

    Parameters:
        measured_twotheta_chi (ndarray): [twotheta, chi] angles in degrees (N x 2).
        B_matrix (ndarray): Original B matrix (3x3).
        hkls (ndarray): Miller indices (N x 3).
        initial_euler_angles (tuple, optional): Initial Euler angles [alpha, beta, gamma] in degrees.

    Returns:
        tuple:
            optimized_R_matrix (ndarray): Optimized orientation matrix (3x3).
            optimized_euler_angles (ndarray): Optimized Euler angles in degrees (3,).
            final_residuals (ndarray): Angular residuals in radians (N,).
            info (dict): Optimization info.
    """
    # Convert measured angles to unit directions
    measured_directions = convert_angle_to_direction(measured_twotheta_chi)

    def residual_func(euler_angles):
        return residuals_orientation(euler_angles, measured_directions, B_matrix, hkls)
    
    # Optimize Euler angles only
    optimized_params, cov_x, infodict, mesg, ier = leastsq(
        residual_func,
        initial_euler_angles,
        full_output=True,
        xtol=1.0e-8,
        ftol=1.0e-8,
        maxfev=10000
    )

    # Compute final residuals
    final_residuals = residuals_orientation(optimized_params, measured_directions, B_matrix, hkls)

    # Convert Euler angles to rotation matrix
    optimized_rotation = R.from_euler('xyz', optimized_params, degrees=True).as_matrix()

    # Compile optimization info
    info = {
        'ier': ier,
        'mesg': mesg,
        'nfev': infodict['nfev'],
        'cov_x': cov_x
    }

    # Return orientation matrix first, then Euler angles, final residuals, and info
    return optimized_rotation, optimized_params, final_residuals, info

def optimize_orientation_and_strain(measured_twotheta_chi, B_matrix, hkls, initial_params=None):
    """
    Optimize both orientation (Euler angles) and deviatoric strain parameters.

    Parameters:
        measured_twotheta_chi (ndarray): [twotheta, chi] angles in degrees (N x 2).
        B_matrix (ndarray): Original B matrix (3x3).
        hkls (ndarray): Miller indices (N x 3).
        initial_params (array-like, optional): Initial guess for [alpha, beta, gamma, E11, E22, E12, E13, E23].
            If None, defaults to zero strain and no rotation.

    Returns:
        tuple:
            optimized_params (ndarray): Optimized parameters [alpha, beta, gamma, E11, E22, E12, E13, E23].
            final_residuals (ndarray): Angular residuals in radians.
            info (dict): Optimization info.
    """
    # Convert measured angles to unit directions
    measured_directions = convert_angle_to_direction(measured_twotheta_chi)

    # Default initial parameters if not provided
    # [alpha, beta, gamma, E11, E22, E12, E13, E23]
    if initial_params is None:
        initial_params = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def residual_func(p):
        return residuals_orientation_and_strain(p, measured_directions, B_matrix, hkls)
    
    # Optimize orientation and strain
    optimized_params, cov_x, infodict, mesg, ier = leastsq(
        residual_func,
        initial_params,
        full_output=True,
        xtol=1.0e-11,
        maxfev=10000
    )

    final_residuals = residuals_orientation_and_strain(optimized_params, measured_directions, B_matrix, hkls)
    info = {
        'ier': ier,
        'mesg': mesg,
        'nfev': infodict['nfev'],
        'cov_x': cov_x
    }

    return optimized_params, final_residuals, info
# Example usage:
# B_matrix = np.array(...)  # Your B matrix for the material
# hkls = np.array([[1,0,0], [0,1,0], [0,0,1], ...])  # List of hkl values as a Nx3 array
# measured_twotheta_chi = np.array([[twotheta1, chi1], [twotheta2, chi2], ...])

# Perform optimization
# optimized_R_matrix, result = optimize_R_matrix(measured_twotheta_chi, B_matrix, hkls)
# print("Optimized R matrix:\n", optimized_R_matrix)
# print("Optimization success:", result.success)
# print("Final angular deviation:", result.fun)
