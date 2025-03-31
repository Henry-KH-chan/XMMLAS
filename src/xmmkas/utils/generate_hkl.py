# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 11:35:10 2024

@author: KHChan
"""

import sys
sys.path.append('..')
import numpy as np
import concurrent.futures

# Import functions and classes from LaueTools and local modules.
from LaueTools.CrystalParameters import Prepare_Grain
from LaueTools.lauecore import getLaueSpots, filterLaueSpots
from LaueTools.dict_LaueTools import dict_Materials
import copy

try:
    # Try relative import first.
    from .Laue_pattern import LauePattern, Symmetry
    from .rotation_transformation import quaternion_to_matrix, rotation_matrix_z, rotation_matrix_y, rotation_matrix_x
    from .twicetheta2pixel import twothetachi_to_pixel_XMAS
    from .pixel22theta import convert_pixel_to_theta_chi_XMAS
except ImportError:
    # Fall back to absolute import if the relative import fails.
    from Laue_pattern import LauePattern, Symmetry
    from rotation_transformation import quaternion_to_matrix, rotation_matrix_z, rotation_matrix_y, rotation_matrix_x
    from twicetheta2pixel import twothetachi_to_pixel_XMAS
    from pixel22theta import convert_pixel_to_theta_chi_XMAS


def simulate_laue_spots(orient_matrix, key_material="BaTiO3", detector_distance=151.094,
                          detector_diameter=200 * 1.717, pixel_size=0.1716):
    """
    Simulate Laue spots for a given orientation matrix and key material.
    
    Parameters:
        orient_matrix (np.array): A 3x3 orientation matrix.
        key_material (str): Material key for simulation. Defaults to "BaTiO3".
        detector_distance (float): Distance from the sample to the detector.
        detector_diameter (float): Effective detector diameter.
        pixel_size (float): Pixel size of the detector.
    
    Returns:
        tuple: (twothetachi, hkl), where:
            - twothetachi is an array of [two theta, chi] pairs.
            - hkl is an array of corresponding Miller indices.
    """
    # Prepare the grain based on the provided material and orientation.
    grain = Prepare_Grain(key_material, orient_matrix, dictmaterials=dict_Materials)
    
    # Simulate Laue spots. The values 0.619920965 and 2.479683325026792 seem to be fixed parameters;
    # consider parameterizing these if they might change.
    spots2pi = getLaueSpots(0.619920965, 2.479683325026792, [grain], dictmaterials=dict_Materials,
                            kf_direction='Z>0')
    # Filter the spots based on geometry and detector parameters.
    twotheta, chi, hkl = filterLaueSpots(spots2pi, fileOK=0, detectordistance=detector_distance,
                                          detectordiameter=detector_diameter, fastcompute=1, pixelsize=pixel_size)
    # Combine two theta and chi into a single array.
    twothetachi = np.vstack((twotheta, chi)).T
    return twothetachi, hkl


def calculate_d_spacing(B_matrix, hkl):
    """
    Calculate the d-spacing for given Miller indices using the B-matrix.
    
    Parameters:
        B_matrix (np.array): B-matrix of the crystal.
        hkl (np.array): Miller indices as a vector.
    
    Returns:
        float: The d-spacing calculated from the norm of the reciprocal lattice vector.
    """
    # Multiply the B-matrix by the hkl vector to obtain the reciprocal lattice vector.
    G_vector = B_matrix @ hkl
    # d-spacing is the inverse of the norm of G_vector.
    d_spacing = 1 / np.linalg.norm(G_vector, axis=0)
    return d_spacing


def convert_quaternion_signs(quat):
    """
    Adjust the signs of a quaternion's components based on the product of the i and j components.
    
    Parameters:
        quat (list or tuple): Quaternion in the form [w, i, j, k].
    
    Returns:
        list: Adjusted quaternion.
    
    Note:
        The function flips the sign of j or i depending on their product, and always flips the sign of k.
    """
    w, i, j, k = quat
    ij_product = i * j

    if ij_product < 0:
        j *= -1  # Keep i, flip j
    elif ij_product > 0:
        i *= -1  # Flip i, keep j

    # Always flip the sign of k.
    k *= -1
    return [w, j, i, k]


def filter_spots_within_view(twicethetachi, hkl, camera_dim, camera_size, sample_detector_distance, center_channel, tilt):
    """
    Filters Laue spots so that only those within the defined camera view are retained.
    
    Parameters:
        twicethetachi (np.array): Array of [two theta, chi] values.
        hkl (np.array): Array of Miller indices.
        camera_dim (tuple): Dimensions of the camera (width, height).
        camera_size (tuple): Physical size of the camera sensor.
        sample_detector_distance (float): Distance from the sample to the detector.
        center_channel (tuple): Center channel coordinates.
        tilt (tuple): Tilt parameters of the detector.
    
    Returns:
        tuple: (filtered_twothetachi, filtered_hkl) for spots within the view.
    """
    # Convert twothetachi values to pixel coordinates using XMAS conventions.
    [x_pix, y_pix] = twothetachi_to_pixel_XMAS(twicethetachi, camera_dim, camera_size, sample_detector_distance,
                                               center_channel, tilt)
    # Define pixel bounds (0-indexed).
    x_min, x_max = 0, 1042  # 1043 pixels in width.
    y_min, y_max = 0, 980   # 981 pixels in height.
    
    ind = (x_pix >= x_min) & (x_pix <= x_max) & (y_pix >= y_min) & (y_pix <= y_max)
    # Round the pixel coordinates and convert back to (two theta, chi) values.
    twicethetachi = convert_pixel_to_theta_chi_XMAS(3 * np.round(x_pix / 3), 3 * np.round(y_pix / 3),
                                                     camera_dim, camera_size, sample_detector_distance,
                                                     center_channel, tilt)
    twicethetachi = np.array(twicethetachi).T
    
    # Apply the boolean index to filter both the twicethetachi and hkl arrays.
    spots_within_view = twicethetachi[ind], hkl[ind]
    return spots_within_view


def execute_simulations(rotmatrices, key_material="BaTiO3"):
    """
    Executes Laue spot simulations concurrently for a list of rotation matrices.
    
    Parameters:
        rotmatrices (list): List of 3x3 rotation matrices.
        key_material (str): Material key for simulation.
    
    Returns:
        list: Simulation results for each rotation matrix.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        # Submit simulation tasks concurrently. Each task applies a rotation matrix 
        # (multiplied with a rotation about Y-axis, for example) to simulate spots.
        futures = {executor.submit(simulate_laue_spots, np.matmul(rotation_matrix_y(0), mat), key_material=key_material): i 
                   for i, mat in enumerate(rotmatrices)}
        results = [None] * len(rotmatrices)
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            results[index] = future.result()
    return results


def process_laue_spots(twicetheta_chi, camera_params, symmetry=Symmetry.tetragonal):
    """
    Processes a list of simulated Laue spot data to generate LauePattern objects.
    
    Parameters:
        twicetheta_chi (list): List of tuples, each containing (twothetachi, hkl) arrays.
        camera_params (tuple): Camera configuration parameters.
        symmetry (Symmetry): Symmetry group to use for the LauePattern.
    
    Returns:
        list: A list of LauePattern objects generated from the input data.
    """
    lauepattern = []
    for twothetachi, hkl in twicetheta_chi:
        twothetachi, hkl_indices = filter_spots_within_view(twothetachi, hkl, *camera_params)
        lauepattern.append(LauePattern(twothetachi, hkl_indices, symmetry_group=symmetry))
    return lauepattern


def generate_laue_hkl(key_material="BaTiO3", valid=False, symmetry=Symmetry.tetragonal):
    """
    Generates a list of LauePattern objects with associated HKL data.
    
    Parameters:
        key_material (str): Material key for simulation.
        valid (bool): Flag to select a subset of uniform orientations.
                      If True, uses one set; if False, uses another set.
        symmetry (Symmetry): Symmetry group for the Laue patterns.
    
    Returns:
        list: A list of LauePattern objects.
    """
    # Load rotation matrices from a precomputed npz file.
    # Note: The file 'uniform_orientations_2000.npz' should contain at least two arrays.
    if valid:
        rotmatrices = np.load('uniform_orientations_2000.npz')['arr_1'][:500]
    else:
        rotmatrices = np.load('uniform_orientations_2000.npz')['arr_0'][:500]
    
    # Execute simulations concurrently to generate Laue spot data.
    results = execute_simulations(rotmatrices, key_material=key_material)
    
    # Define camera parameters: (camera dimensions, camera size, detector distance, center channel, tilt).
    camera_params = ((1043, 981), (179, 168.387), 151.094, (528.471, 198.304), (0, -0.281, 0.592))
    
    # Process the simulation results to produce LauePattern objects.
    lauepattern = process_laue_spots(results, camera_params, symmetry=symmetry)
    return lauepattern

# Uncomment the following block to run a quick test simulation when this module is executed directly.
if __name__ == "__main__":
    laue_patterns = generate_laue_hkl()
    print("Generated", len(laue_patterns), "Laue patterns.")
