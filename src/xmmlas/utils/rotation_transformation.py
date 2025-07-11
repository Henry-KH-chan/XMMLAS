# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 11:39:11 2024

@author: KHChan
"""

import numpy as np


def quaternion_to_matrix(quaternion):
    """
    Convert a quaternion into a 3x3 rotation matrix.

    Parameters:
    quaternion (array-like): Quaternion in the format [q0, qx, qy, qz],
                             where q0 is the scalar part, and (qx, qy, qz)
                             are the vector part.

    Returns:
    numpy.ndarray: 3x3 rotation matrix.
    """
    q0, qx, qy, qz = quaternion
    # Normalize the quaternion to ensure the rotation matrix is valid
    norm = np.sqrt(q0 * q0 + qx * qx + qy * qy + qz * qz)
    q0, qx, qy, qz = q0 / norm, qx / norm, qy / norm, qz / norm

    # Compute the rotation matrix elements
    r11 = 2 * q0 * q0 - 1 + 2 * qx * qx
    r12 = 2 * (qx * qy - q0 * qz)
    r13 = 2 * (qx * qz + q0 * qy)
    r21 = 2 * (qx * qy + q0 * qz)
    r22 = 2 * q0 * q0 - 1 + 2 * qy * qy
    r23 = 2 * (qy * qz - q0 * qx)
    r31 = 2 * (qx * qz - q0 * qy)
    r32 = 2 * (qy * qz + q0 * qx)
    r33 = 2 * q0 * q0 - 1 + 2 * qz * qz

    # Construct the rotation matrix
    rotation_matrix = np.array([[r11, r12, r13],
                                [r21, r22, r23],
                                [r31, r32, r33]])

    return rotation_matrix


def matrix_to_quaternion(matrix):
    """
    Convert a 3x3 rotation matrix to a quaternion.

    Parameters:
    matrix (numpy.ndarray): 3x3 rotation matrix.

    Returns:
    numpy.ndarray: Quaternion in the format [q0, qx, qy, qz].
    """
    m = matrix
    trace = np.trace(m)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        q0 = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q0 = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q0 = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q0 = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s

    quaternion = np.array([q0, qx, qy, qz])
    # Normalize the quaternion to ensure its magnitude is 1
    quaternion /= np.linalg.norm(quaternion)
    return quaternion


def rotation_matrix_x(angle_degrees):
    angle_radians = np.radians(angle_degrees)
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)
    return np.array([[1, 0, 0],
                     [0, cos_theta, -sin_theta],
                     [0, sin_theta, cos_theta]])


def rotation_matrix_y(angle_degrees):
    # Convert degrees to radians
    rad = np.radians(angle_degrees)
    # Create the rotation matrix
    R_y = np.array([[np.cos(rad), 0, np.sin(rad)],
                    [0, 1, 0],
                    [-np.sin(rad), 0, np.cos(rad)]])
    return R_y


def rotation_matrix_z(angle_degrees):
    # Convert angle from degrees to radians
    angle_radians = np.radians(angle_degrees)

    # Define the rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians), 0],
        [np.sin(angle_radians), np.cos(angle_radians), 0],
        [0, 0, 1]
    ])

    return rotation_matrix


def euler_to_rotation_matrix(roll, pitch, yaw):
    # Convert Euler angles (in degrees) to rotation matrix
    # The rotation order is assumed to be ZYX (yaw, pitch, roll)

    # Convert angles from degrees to radians
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)

    # Calculate rotation matrices
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])

    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])

    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

    # Combined rotation matrix
    R = np.dot(R_z, np.dot(R_y, R_x))

    return R

def rotation_matrix_to_euler_xyz(R):
    ψ = np.arctan2(R[1, 0], R[0, 0])
    θ = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    φ = np.arctan2(R[2, 1], R[2, 2])
    return np.array([φ, θ, ψ])