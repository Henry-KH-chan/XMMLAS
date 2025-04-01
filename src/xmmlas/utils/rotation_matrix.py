# -*- coding: utf-8 -*- 
"""
Created on Fri May 17 11:25:46 2024

@author: KHChan
"""

import numpy as np
DEG = 180/np.pi  # Conversion factor from radians to degrees

try:
    # Try relative import first for modular usage
    from .distance import compute_angular_distance, calculate_angles_between_hkls, compute_angular_distance2
except ImportError:
    # Fall back to absolute import if relative import fails
    from distance import compute_angular_distance, calculate_angles_between_hkls, compute_angular_distance2


def normalize(vector):
    """ Normalize a vector. """
    return vector / np.linalg.norm(vector)


def rotation_matrix_about_axis(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians using Rodrigues' rotation formula.

    Args:
        axis (array-like): The rotation axis.
        theta (float): Rotation angle in radians.

    Returns:
        np.array: A 3x3 rotation matrix.
    """
    # Normalize the axis vector
    axis = np.asarray(axis)
    axis = normalize(axis)
    
    # Construct the skew-symmetric matrix for the axis.
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    
    # Compute the rotation matrix using Rodrigues' formula.
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    
    return R


def convert_angle_to_direction(twotheta_deg_chi_deg):
    """
    Convert (two theta, chi) angles in degrees to a 3D direction vector.
    
    The two theta value is halved and converted to radians, while chi is converted directly.

    Args:
        twotheta_deg_chi_deg (array-like): Array of [two theta, chi] in degrees.

    Returns:
        np.array: 3D vector (x, y, z) corresponding to the angles.
    """
    twotheta_deg_chi_deg = np.array(twotheta_deg_chi_deg)
    theta_deg = twotheta_deg_chi_deg.T[0] / 2  # Halve the two theta value
    chi_deg = twotheta_deg_chi_deg.T[1]
    
    theta_rad = theta_deg / DEG  # Convert to radians
    chi_rad = chi_deg / DEG      # Convert to radians
    
    # Compute the direction vector components.
    x = -np.sin(theta_rad)
    y = np.cos(theta_rad) * np.sin(chi_rad)
    z = np.cos(theta_rad) * np.cos(chi_rad)
    
    return np.array([x, y, z])


def convert_direction_to_angle(xyz):
    """
    Convert a 3D direction vector to (two theta, chi) angles in degrees.
    
    Args:
        xyz (array-like): 3D vector (x, y, z).
    
    Returns:
        np.array: Array containing [two theta, chi] in degrees.
    """
    xyz = normalize(np.array(xyz))
    x = xyz.T[0]
    y = xyz.T[1]
    z = xyz.T[2]
    
    theta_rad = np.arcsin(-x)
    chi_rad = np.arctan(y / z)
    
    theta_deg = theta_rad * DEG
    chi_deg = chi_rad * DEG
        
    return np.array([theta_deg * 2, chi_deg])  # two theta is twice theta


def compute_rotation_matrix(points, hkls, B_matrix):
    """
    Compute a rotation matrix that aligns the predicted Laue pattern with the experimental data.
    
    This function uses two HKL vectors to determine the best rotation matrix by:
        1. Computing the first rotation from the first HKL.
        2. Using the resulting transformed HKLs to compute a second rotation.
    
    Args:
        points (np.array): Array of experimental [two theta, chi] points.
        hkls (list or np.array): List/array of HKL vectors.
        B_matrix (np.array): The B-matrix used for transformation.
    
    Returns:
        np.array: A 3x3 rotation matrix.
    """
    hkl1 = hkls[0]
    point1 = points[0]
    rec_vec1 = normalize(np.dot(B_matrix, hkl1))
    
    diff_vec1 = convert_angle_to_direction(point1)
    first_axis = np.cross(rec_vec1, diff_vec1)
    rot_angle1 = calculate_angles_between_hkls([rec_vec1, diff_vec1])[0, 1]
    
    first_rotation_matrix = rotation_matrix_about_axis(first_axis, rot_angle1 / DEG)
    
    new_points = [convert_direction_to_angle(np.dot(first_rotation_matrix, np.dot(B_matrix, hkl))) for hkl in hkls]
    angular_difference = compute_angular_distance2(np.array(new_points), points)
    index = np.argmin(angular_difference[1:]) + 1
    hkl2 = hkls[index]
    
    point2 = points[index]
    rec_vec2 = normalize(np.dot(B_matrix, hkl2))
    diff_vec2 = convert_angle_to_direction(point2)
    
    rec_vec2 = np.dot(first_rotation_matrix, rec_vec2)
    rot_angle2 = calculate_angles_between_hkls([np.cross(diff_vec1, rec_vec2), np.cross(diff_vec1, diff_vec2)])[0, 1]
    
    second_rotation_matrix = rotation_matrix_about_axis(diff_vec1, -rot_angle2 / DEG)
    
    return np.dot(second_rotation_matrix, first_rotation_matrix)


def compute_first_rotation_matrix(point, hkl, Bmatrix):
    """
    Compute the first rotation matrix needed to align a single HKL vector with an experimental point.
    
    Args:
        point (array-like): Experimental [two theta, chi] angles.
        hkl (array-like): HKL vector.
        Bmatrix (np.array): B-matrix.
    
    Returns:
        np.array: A 3x3 rotation matrix computed from the first HKL and point.
    """
    hkl1 = hkl
    point1 = point
    rec_vec1 = normalize(np.dot(Bmatrix, hkl))
    diff_vec1 = convert_angle_to_direction(point1)
    first_axis = np.cross(rec_vec1, diff_vec1)
    rot_angle1 = calculate_angles_between_hkls([rec_vec1, diff_vec1])[0, 1]
    first_rotation_matrix = rotation_matrix_about_axis(first_axis, rot_angle1 / DEG)
    return first_rotation_matrix

# Example test code (commented out):
"""
# Example usage of compute_rotation_matrix:
# Define sample HKLs and experimental points.
hkl1 = [-1, 0, 1]
hkl2 = [-3, 0, 5]
point1 = [55.03, 11.72]
point2 = [63.58, -3.72]

# Bmatrix for the crystal (example values)
Bmatrix = np.array([
    [0.2505, 0, 0],
    [0, 0.2505, 0],
    [0, 0, 0.2478]
])

# Example call to compute_rotation_matrix (you would need a full set of points and hkls).
R = compute_rotation_matrix(np.array([point1, point2]), [hkl1, hkl2], Bmatrix)
print("Computed Rotation Matrix:\n", R)
"""
