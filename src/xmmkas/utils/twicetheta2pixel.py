# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:25:41 2024

@author: KHChan
"""

import numpy as np
from scipy.optimize import minimize

def twothetachi_to_pixel_XMAS(twothetachi, camera_dim, camera_size, sample_detector_distance, center_channel, tilt):
    """
    Convert (two theta, chi) angles in degrees to pixel coordinates using the XMAS convention.
    
    Parameters:
        twothetachi (np.array): Array of [two theta, chi] angles in degrees with shape (N, 2).
        camera_dim (tuple): Dimensions of the CCD camera in pixels (width, height).
        camera_size (tuple): Physical size of the CCD camera in mm (width, height).
        sample_detector_distance (float): Distance from the sample to the detector in mm.
        center_channel (tuple): Center channel position in pixels (x, y).
        tilt (tuple): Detector tilt (roll, pitch, yaw) in degrees.
    
    Returns:
        tuple: (x_pix, y_pix) arrays of pixel coordinates.
    """
    # Convert parameters to numpy arrays.
    camera_dim_pix = np.array(camera_dim)     # (width, height) in pixels
    camera_size_mm = np.array(camera_size)      # (width, height) in mm
    center_channel_pix = np.array(center_channel)  # Center pixel coordinates
    
    # Convert tilt angles from degrees to radians.
    roll, pitch, yaw = np.radians(tilt)
    # Convert two theta and chi angles from degrees to radians.
    twothetachi = np.radians(twothetachi)

    # Compute the pixel size in mm (assuming square pixels if not otherwise).
    pixel_size = camera_size_mm / camera_dim_pix

    # Pre-calculate trigonometric terms.
    sinchi = np.sin(twothetachi[:, 1])
    coschi = np.cos(twothetachi[:, 1])
    tantwotheta = np.tan(twothetachi[:, 0])
    sinpitch = np.sin(pitch)
    cospitch = np.cos(pitch)
    sinyaw = np.sin(yaw)
    cosyaw = np.cos(yaw)

    # Calculate radial distance (r) using the provided geometric relations.
    # Note: The equation below assumes a specific geometry; ensure that it matches your experimental setup.
    r = sample_detector_distance * cospitch / (
                sinchi * sinyaw * sinpitch + coschi * cospitch - cosyaw * sinpitch / tantwotheta)

    # Compute x, y, z coordinates in the laboratory frame (in mm)
    x_mm = -r * sinchi
    y_mm = r / tantwotheta
    z_mm = r * coschi

    # Convert z coordinate to the detector frame (in mm)
    z_mm_detector = r * coschi - sample_detector_distance

    # Compute y coordinate in the detector frame
    y_mm_detector = z_mm_detector / sinpitch
    # Compute x coordinate in the detector frame using tilt corrections.
    x_mm_detector = (y_mm_detector * cosyaw * cospitch - y_mm) / sinyaw

    # Convert distances in mm to pixel coordinates relative to the center of the detector.
    x_pix = x_mm_detector / pixel_size[0] + center_channel_pix[0]
    y_pix = y_mm_detector / pixel_size[1] + center_channel_pix[1]
    return x_pix, y_pix

"""
# Example test code:
x = 1  # Example pixel value (you might want to provide a full array if testing multiple points)
y = 1  # Example pixel value
two_theta = 100.57063588339366
chi = 30.995433433044862
camera_dim = (1043, 981)
camera_size = (179, 168.387)
sample_detector_distance = 151.094  # in mm
center_channel = (528.471, 198.304)  # in pixels
tilt = (0, -0.281, 0.592)  # (roll, pitch, yaw) in degrees

# Construct a test input: For a single point, twothetachi should be a 2D array.
twothetachi_test = np.array([[two_theta, chi]])
x_pix, y_pix = twothetachi_to_pixel_XMAS(twothetachi_test, camera_dim, camera_size, sample_detector_distance, center_channel, tilt)
print("x_pix:", x_pix)
print("y_pix:", y_pix)
"""
