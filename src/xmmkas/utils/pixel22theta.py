# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 15:22:16 2023

@author: KHChan
"""

import numpy as np

def convert_pixel_to_theta_chi_XMAS(x_pix, y_pix, camera_dim, camera_size, sample_detector_distance, center_channel, tilt):
    """
    Convert pixel coordinates to (two theta, chi) angles using the XMAS convention.
    
    Parameters:
        x_pix (float or np.array): Pixel coordinate(s) in the x-direction.
        y_pix (float or np.array): Pixel coordinate(s) in the y-direction.
        camera_dim (tuple): Dimensions of the camera in pixels (width, height).
        camera_size (tuple): Physical size of the camera in mm (width, height).
        sample_detector_distance (float): Distance from the sample to the detector in mm.
        center_channel (tuple): Center channel position in pixels (x, y).
        tilt (tuple): Detector tilt in degrees as (roll, pitch, yaw).
    
    Returns:
        tuple: (two_theta_deg, chi_deg) angles in degrees.
    """
    # Convert camera dimensions and size to numpy arrays
    camera_dim_pix = np.array(camera_dim)    # (width, height) in pixels
    camera_size_mm = np.array(camera_size)     # (width, height) in mm
    center_channel_pix = np.array(center_channel)  # (x, y) center in pixels
    
    # Convert tilt from degrees to radians; note the negative sign for yaw as used.
    roll, pitch, yaw = np.radians(tilt)
    
    # Calculate the effective pixel size in mm
    pixel_size = camera_size_mm / camera_dim_pix

    # Convert pixel coordinates to mm relative to the center of the detector
    x_mm_detector = (x_pix - center_channel_pix[0]) * pixel_size[0]
    y_mm_detector = (y_pix - center_channel_pix[1]) * pixel_size[1]
    
    # Apply tilt corrections: rotate the detector coordinates.
    # Adjust x_mm and y_mm based on yaw and pitch.
    x_mm = x_mm_detector * np.cos(-yaw) - y_mm_detector * np.sin(-yaw) * np.cos(pitch)
    y_mm = x_mm_detector * np.sin(-yaw) + y_mm_detector * np.cos(-yaw) * np.cos(pitch)
    z_mm = y_mm_detector * np.sin(pitch)
    
    # Compute the magnitude of the vector from the sample to the pixel (including detector distance)
    mag = np.sqrt(x_mm**2 + y_mm**2 + (sample_detector_distance + z_mm)**2)
    
    # Calculate two_theta angle: using arccos of y_mm/mag.
    two_theta_rad = np.arccos(y_mm / mag)
    two_theta_deg = np.degrees(two_theta_rad)
    
    # Calculate chi: angle in the detector plane using arctan.
    chi_rad = np.arctan(-x_mm / (z_mm + sample_detector_distance))
    chi_deg = np.degrees(chi_rad)
    
    return two_theta_deg, chi_deg


def convert_pixel_to_theta_chi_XMAS2(x_pix, y_pix, camera_dim, camera_size, sample_detector_distance, center_channel, tilt, angle_rad=90):
    """
    Alternate conversion of pixel coordinates to (two theta, chi) angles.
    
    Uses explicit rotation matrices for yaw, roll, and pitch, and applies a final rotation
    defined by angle_rad. This function uses a fixed pixel size.
    
    Parameters:
        x_pix (float or np.array): Pixel coordinate(s) in x-direction.
        y_pix (float or np.array): Pixel coordinate(s) in y-direction.
        camera_dim (tuple): Dimensions of the camera in pixels.
        camera_size (tuple): Size of the camera in mm.
        sample_detector_distance (float): Detector distance in mm.
        center_channel (tuple): Center channel position in pixels.
        tilt (tuple): Detector tilt (roll, pitch, yaw) in degrees.
        angle_rad (float): Additional rotation angle in degrees. Default is 90.
    
    Returns:
        tuple: (two_theta_deg, chi_deg) angles in degrees.
    """
    camera_dim_pix = np.array(camera_dim)
    camera_size_mm = np.array(camera_size)
    center_channel_pix = np.array(center_channel)
    roll, pitch, yaw = np.radians(tilt)
    # Here pixel size is fixed (overriding camera_size and camera_dim)
    pixel_size = camera_size_mm / camera_dim_pix
    angle_rad = np.radians(angle_rad)

    # Convert pixel coordinates to mm relative to the center of the detector.
    x_mm_detector = (x_pix - center_channel_pix[0]) * pixel_size
    y_mm_detector = (y_pix - center_channel_pix[1]) * pixel_size
    
    # Define rotation matrices for yaw, roll, and pitch.
    R_yaw = np.array([
        [np.cos(-yaw), -np.sin(-yaw), 0],
        [np.sin(-yaw),  np.cos(-yaw), 0],
        [0, 0, 1]
    ])
    
    R_roll = np.array([
        [np.cos(roll), 0, np.sin(roll)],
        [0, 1, 0],
        [-np.sin(roll), 0, np.cos(roll)]
    ])
    
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch),  np.cos(pitch)]
    ])
    
    # Additional rotation (around x-axis) by angle_rad
    R_detector = np.array([
        [1, 0, 0],
        [0, np.cos(angle_rad), -np.sin(angle_rad)],
        [0, np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    
    # Combine rotations for detector tilt corrections.
    R_combined = R_yaw @ R_roll @ R_pitch
    
    # Construct direction vectors from the pixel coordinates (set z to 0 initially).
    direction_vectors = np.vstack((x_mm_detector, y_mm_detector, np.zeros(y_mm_detector.shape))).T  # shape: (N, 3)
    
    # Apply combined rotations to these vectors.
    rotated_vectors = R_combined @ direction_vectors.T
    # Add the detector distance in the z direction.
    rotated_vectors[2] = rotated_vectors[2] + sample_detector_distance
    # Apply the final detector rotation.
    rotated_vectors = R_detector @ rotated_vectors
    x_mm = rotated_vectors.T[:, 0]
    y_mm = rotated_vectors.T[:, 1]
    z_mm = rotated_vectors.T[:, 2]
    
    mag = np.sqrt(x_mm**2 + y_mm**2 + z_mm**2)
    two_theta_rad = np.arccos(z_mm / mag)
    two_theta_deg = np.degrees(two_theta_rad)
    
    # Compute chi angle; note the use of arccos and the sign of x_mm for determining quadrant.
    chi_rad = -np.sign(x_mm) * np.arccos(-y_mm / np.sqrt(x_mm**2 + y_mm**2))
    chi_deg = np.degrees(chi_rad)
    
    return two_theta_deg, chi_deg


# Test code (commented out):
"""
# Example test for convert_pixel_to_theta_chi_XMAS
x = 1043  # x position in pixels
y = 981   # y position in pixels
camera_dim = (1043, 981)
camera_size = (179, 168.387)
sample_detector_distance = 151.094  # in mm
center_channel = (528.471, 198.304)
tilt = (0, -0.281, 0.592)  # tilt: (roll, pitch, yaw) in degrees

two_theta, chi = convert_pixel_to_theta_chi_XMAS(x, y, camera_dim, camera_size, sample_detector_distance, center_channel, tilt)
print("2θ angle:", two_theta)
print("chi angle:", chi)
"""
