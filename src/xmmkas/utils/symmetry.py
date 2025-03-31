# -*- coding: utf-8 -*-
"""
Created on Tue May 14 23:32:21 2024

@author: KHChan
"""

import numpy as np
import enum
import logging
import itertools
from math import sin, radians

# Define an enumeration for crystal symmetries.
class Symmetry(enum.Enum):
    cubic = 'm3m'
    hexagonal = '6/mmm'
    orthorhombic = 'mmm'
    tetragonal = '4/mmm'
    trigonal = 'bar3m'
    monoclinic = '2/m'
    triclinic = 'bar1'
    
    @staticmethod
    def from_string(s):
        """
        Convert a string to the corresponding Symmetry enum member.
        Raises a ValueError if the provided string is not supported.
        """
        for e in Symmetry:
            if e.value == s:
                return e
        raise ValueError(f"Unsupported symmetry value: {s}")
        
    def get_relevant_lattice_params(self, full_params):
        """
        Extract relevant lattice parameters based on the symmetry.
        
        Parameters:
            full_params: tuple or list of all six lattice parameters (a, b, c, alpha, beta, gamma)
        
        Returns:
            A tuple of the lattice parameters relevant for the given symmetry.
        """
        if self == Symmetry.cubic:
            return (full_params[0],)  # For cubic, only parameter 'a' is needed.
        elif self in [Symmetry.tetragonal, Symmetry.hexagonal, Symmetry.trigonal]:
            return (full_params[0], full_params[2])  # Use 'a' and 'c'
        elif self == Symmetry.orthorhombic:
            return (full_params[0], full_params[1], full_params[2])  # a, b, c
        elif self == Symmetry.monoclinic:
            return (full_params[0], full_params[1], full_params[2], full_params[4])  # a, b, c, beta
        elif self == Symmetry.triclinic:
            return full_params[:6]  # a, b, c, alpha, beta, gamma
        else:
            raise NotImplementedError(f"Symmetry '{self}' is not implemented for lattice parameter extraction.")

def apply_symmetry(hkls, symmetry):
    """
    Apply the appropriate symmetry operations to a list/array of Miller indices.
    
    Parameters:
        hkls (array-like): List or array of Miller indices.
        symmetry: Either a string or a Symmetry enum instance.
    
    Returns:
        np.ndarray: Array of symmetrized Miller indices.
    """
    if isinstance(symmetry, str):
        symmetry = Symmetry.from_string(symmetry)
        
    if symmetry._value_ == Symmetry.monoclinic._value_:
        return apply_monoclinic_symmetry(hkls)
    elif symmetry._value_ == Symmetry.triclinic._value_:
        return apply_triclinic_symmetry(hkls)
    elif symmetry._value_ == Symmetry.trigonal._value_:
        return apply_trigonal_symmetry(hkls)
    elif symmetry._value_ == Symmetry.tetragonal._value_:
        return apply_tetragonal_symmetry(hkls)
    elif symmetry._value_ == Symmetry.orthorhombic._value_:
        return apply_orthorhombic_symmetry(hkls)
    elif symmetry._value_ == Symmetry.cubic._value_:
        return apply_cubic_symmetry(hkls)
    else:
        raise ValueError(f"Unsupported symmetry: {symmetry}")

def generate_symmetric_indices(hkl, symmetry):
    """
    Generate all symmetrically equivalent Miller indices for a given HKL vector
    and symmetry.
    
    Parameters:
        hkl (array-like): A Miller index vector.
        symmetry: Either a string or a Symmetry enum instance.
    
    Returns:
        np.ndarray: Array of symmetrically equivalent Miller indices.
    """
    if isinstance(symmetry, str):
        symmetry = Symmetry.from_string(symmetry)

    logging.debug(f"Received symmetry: {symmetry}, type: {type(symmetry)}")
    if symmetry._value_ == Symmetry.cubic._value_:
        return generate_cubic_symmetric_indices(hkl)
    elif symmetry._value_ == Symmetry.trigonal._value_:
        return generate_trigonal_symmetric_indices(hkl)
    elif symmetry._value_ == Symmetry.tetragonal._value_:
        return generate_tetragonal_symmetric_indices(hkl)
    elif symmetry._value_ == Symmetry.orthorhombic._value_:
        return generate_orthorhombic_symmetric_indices(hkl)
    elif symmetry._value_ == Symmetry.monoclinic._value_:
        return generate_monoclinic_symmetric_indices(hkl)
    elif symmetry._value_ == Symmetry.triclinic._value_:
        return generate_triclinic_symmetric_indices(hkl)
    else:
        raise ValueError(f"Unsupported symmetry: {symmetry}")

def compute_d_spacing(hkls, lattice_params, symmetry):
    """
    Compute theoretical d-spacing based on the given symmetry.
    
    Parameters:
        hkls (array-like): Array of Miller indices, shape (N, 3).
        lattice_params (tuple or list): Lattice parameters relevant to the symmetry.
            Examples:
                - Cubic: (a,)
                - Tetragonal: (a, c)
                - Hexagonal: (a, c)
                - Orthorhombic: (a, b, c)
                - Trigonal: (a, c)
                - Monoclinic: (a, b, c, beta)
                - Triclinic: (a, b, c, alpha, beta, gamma)
        symmetry (Symmetry): An instance of the Symmetry enum.
    
    Returns:
        np.ndarray: Array of d-spacings, shape (N,).
    """
    if not isinstance(symmetry, Symmetry):
        raise TypeError("symmetry must be an instance of the Symmetry enum.")
    
    hkls = np.array(hkls)
    
    if symmetry == Symmetry.cubic:
        return compute_d_cubic(hkls, lattice_params)
    elif symmetry == Symmetry.tetragonal:
        return compute_d_tetragonal(hkls, lattice_params)
    elif symmetry == Symmetry.hexagonal:
        return compute_d_hexagonal(hkls, lattice_params)
    elif symmetry == Symmetry.orthorhombic:
        return compute_d_orthorhombic(hkls, lattice_params)
    elif symmetry == Symmetry.trigonal:
        return compute_d_trigonal(hkls, lattice_params)
    elif symmetry == Symmetry.monoclinic:
        return compute_d_monoclinic(hkls, lattice_params)
    elif symmetry == Symmetry.triclinic:
        return compute_d_triclinic(hkls, lattice_params)
    else:
        raise NotImplementedError(f"Symmetry '{symmetry}' is not implemented for d-spacing computation.")

def compute_d_cubic(hkls, lattice_params):
    """
    Compute d-spacing for cubic symmetry.
    
    Parameters:
        hkls (np.ndarray): Array of Miller indices, shape (N, 3).
        lattice_params (tuple): (a,).
    
    Returns:
        np.ndarray: Array of d-spacings.
    """
    a, = lattice_params
    h, k, l = hkls[:, 0], hkls[:, 1], hkls[:, 2]
    denominator = np.sqrt(h**2 + k**2 + l**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        d_spacing = a / denominator
        d_spacing[denominator == 0] = np.inf
    return d_spacing

def compute_d_tetragonal(hkls, lattice_params):
    """
    Compute d-spacing for tetragonal symmetry.
    
    Parameters:
        hkls (np.ndarray): Array of Miller indices.
        lattice_params (tuple): (a, c).
    
    Returns:
        np.ndarray: Array of d-spacings.
    """
    a, c = lattice_params
    h, k, l = hkls[:, 0], hkls[:, 1], hkls[:, 2]
    denominator = np.sqrt((h**2 + k**2) / a**2 + (l**2) / c**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        d_spacing = 1 / denominator
        d_spacing[denominator == 0] = np.inf
    return d_spacing

def compute_d_hexagonal(hkls, lattice_params):
    """
    Compute d-spacing for hexagonal symmetry.
    
    Parameters:
        hkls (np.ndarray): Array of Miller indices.
        lattice_params (tuple): (a, c).
    
    Returns:
        np.ndarray: Array of d-spacings.
    """
    a, c = lattice_params
    h, k, l = hkls[:, 0], hkls[:, 1], hkls[:, 2]
    denominator = np.sqrt((4/3)*(h**2 + h*k + k**2)/a**2 + l**2/c**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        d_spacing = 1 / denominator
        d_spacing[denominator == 0] = np.inf
    return d_spacing

def compute_d_orthorhombic(hkls, lattice_params):
    """
    Compute d-spacing for orthorhombic symmetry.
    
    Parameters:
        hkls (np.ndarray): Array of Miller indices.
        lattice_params (tuple): (a, b, c).
    
    Returns:
        np.ndarray: Array of d-spacings.
    """
    a, b, c = lattice_params
    h, k, l = hkls[:, 0], hkls[:, 1], hkls[:, 2]
    denominator = np.sqrt((h**2)/a**2 + (k**2)/b**2 + (l**2)/c**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        d_spacing = 1 / denominator
        d_spacing[denominator == 0] = np.inf
    return d_spacing

def compute_d_trigonal(hkls, lattice_params):
    """
    Compute d-spacing for trigonal symmetry.
    
    Parameters:
        hkls (np.ndarray): Array of Miller indices.
        lattice_params (tuple): (a, c).
    
    Returns:
        np.ndarray: Array of d-spacings.
    """
    a, c = lattice_params
    h, k, l = hkls[:, 0], hkls[:, 1], hkls[:, 2]
    denominator = np.sqrt((4/3)*(h**2 + h*k + k**2)/a**2 + l**2/c**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        d_spacing = 1 / denominator
        d_spacing[denominator == 0] = np.inf
    return d_spacing

def compute_d_monoclinic(hkls, lattice_params):
    """
    Compute d-spacing for monoclinic symmetry.
    
    Parameters:
        hkls (np.ndarray): Array of Miller indices.
        lattice_params (tuple): (a, b, c, beta) where beta is in degrees.
    
    Returns:
        np.ndarray: Array of d-spacings.
    """
    a, b, c, beta_deg = lattice_params
    beta_rad = radians(beta_deg)
    h, k, l = hkls[:, 0], hkls[:, 1], hkls[:, 2]
    cos_beta = np.cos(beta_rad)
    denominator = (h**2)/a**2 + (k**2)/b**2 + (l**2)/c**2 - (2 * h * l * cos_beta)/(a**2)
    denominator = np.sqrt(denominator)
    with np.errstate(divide='ignore', invalid='ignore'):
        d_spacing = 1 / denominator
        d_spacing[denominator == 0] = np.inf
    return d_spacing

def compute_d_triclinic(hkls, lattice_params):
    """
    Compute d-spacing for triclinic symmetry.
    
    Parameters:
        hkls (np.ndarray): Array of Miller indices.
        lattice_params (tuple): (a, b, c, alpha, beta, gamma) with angles in degrees.
    
    Returns:
        np.ndarray: Array of d-spacings.
    """
    a, b, c, alpha_deg, beta_deg, gamma_deg = lattice_params
    alpha_rad = radians(alpha_deg)
    beta_rad = radians(beta_deg)
    gamma_rad = radians(gamma_deg)

    # Calculate the volume of the unit cell
    volume = a * b * c * np.sqrt(
        1 - np.cos(alpha_rad)**2 - np.cos(beta_rad)**2 - np.cos(gamma_rad)**2 +
        2 * np.cos(alpha_rad) * np.cos(beta_rad) * np.cos(gamma_rad)
    )

    h, k, l = hkls[:, 0], hkls[:, 1], hkls[:, 2]
    numerator = (
        h**2 * b**2 * c**2 * np.sin(alpha_rad)**2 +
        k**2 * a**2 * c**2 * np.sin(beta_rad)**2 +
        l**2 * a**2 * b**2 * np.sin(gamma_rad)**2 +
        2 * h * k * a * b * c**2 * (np.cos(alpha_rad) * np.cos(beta_rad) - np.cos(gamma_rad)) +
        2 * h * l * a * c * b**2 * (np.cos(alpha_rad) * np.cos(gamma_rad) - np.cos(beta_rad)) +
        2 * k * l * b * c * a**2 * (np.cos(beta_rad) * np.cos(gamma_rad) - np.cos(alpha_rad))
    )
    denominator = volume**2
    with np.errstate(divide='ignore', invalid='ignore'):
        d_spacing = volume / np.sqrt(numerator / denominator)
        d_spacing[np.isnan(d_spacing)] = np.inf
    return d_spacing

def generate_cubic_symmetric_indices(hkl):
    """
    Generate all symmetrically equivalent Miller indices for cubic symmetry in a specific order.
    
    Parameters:
        hkl (tuple or list): Miller indices (h, k, l).
    
    Returns:
        np.ndarray: Array of unique symmetric Miller indices.
    """
    h, k, l = hkl

    # Generate all unique permutations of (h, k, l)
    perms = set(itertools.permutations([h, k, l]))

    unique_indices = set()
    ordered_indices = []

    # Define sign combinations for different numbers of negative signs
    sign_combinations = {
        0: [(1, 1, 1)],
        1: [(-1, 1, 1), (1, -1, 1), (1, 1, -1)],
        2: [(1, -1, -1), (-1, 1, -1), (-1, -1, 1)],
        3: [(-1, -1, -1)],
    }

    # Enforce a specific order based on number of negative signs
    for num_neg in range(0, 4):
        signs_list = sign_combinations[num_neg]
        for perm in perms:
            for signs in signs_list:
                idx = tuple(s * p for s, p in zip(signs, perm))
                if idx not in unique_indices:
                    unique_indices.add(idx)
                    ordered_indices.append(idx)

    return np.array(ordered_indices)

def generate_trigonal_symmetric_indices(hkl):
    """
    Generate all symmetrically equivalent Miller indices for trigonal symmetry.
    
    Parameters:
        hkl (tuple or list): Miller indices (h, k, l).
    
    Returns:
        np.ndarray: Array of unique symmetric Miller indices.
    """
    h, k, l = hkl

    # Three-fold rotational symmetry around the c-axis.
    rotations = [
        (h, k, l),
        (-k, h + k, l),
        (-h - k, h, l)
    ]

    # For centrosymmetric trigonal crystals, include inversion.
    inversions = [(-idx[0], -idx[1], -idx[2]) for idx in rotations]

    all_indices = rotations + inversions
    unique_indices = list(set(all_indices))
    return np.array(unique_indices)

def generate_tetragonal_symmetric_indices(hkl):
    """
    Generate all symmetrically equivalent Miller indices for tetragonal symmetry.
    
    Parameters:
        hkl (tuple or list): Miller indices (h, k, l).
    
    Returns:
        np.ndarray: Array of unique symmetric Miller indices.
    """
    h, k, l = hkl

    rotations = [
        (h, k, l),
        (-k, h, l),
        (-h, -k, l),
        (k, -h, l)
    ]
    inversions = [(-h, -k, -l) for h, k, l in rotations]
    all_indices = rotations + inversions
    unique_indices = list(set(all_indices))
    return np.array(unique_indices)

def generate_monoclinic_symmetric_indices(hkl):
    """
    Generate all symmetrically equivalent Miller indices for monoclinic symmetry.
    
    Parameters:
        hkl (tuple or list): Miller indices (h, k, l).
    
    Returns:
        np.ndarray: Array of unique symmetric Miller indices.
    """
    h, k, l = hkl

    rotations = [
        (h, k, l),
        (-h, k, -l)
    ]
    inversions = [(-h, -k, -l) for h, k, l in rotations]
    all_indices = rotations + inversions
    unique_indices = list(set(all_indices))
    return np.array(unique_indices)

def generate_orthorhombic_symmetric_indices(hkl):
    """
    Generate all symmetrically equivalent Miller indices for orthorhombic symmetry.
    
    Parameters:
        hkl (tuple or list): Miller indices (h, k, l).
    
    Returns:
        np.ndarray: Array of unique symmetric Miller indices.
    """
    h, k, l = hkl

    operations = [
        (h, k, l),
        (-h, k, l),
        (h, -k, l),
        (h, k, -l),
        (-h, -k, l),
        (-h, k, -l),
        (h, -k, -l),
        (-h, -k, -l)
    ]
    unique_indices = list(set(operations))
    return np.array(unique_indices)

def generate_triclinic_symmetric_indices(hkl):
    """
    Generate all symmetrically equivalent Miller indices for triclinic symmetry.
    
    Parameters:
        hkl (tuple or list): Miller indices (h, k, l).
    
    Returns:
        np.ndarray: Array of symmetric Miller indices (identity and inversion).
    """
    h, k, l = hkl
    indices = [(h, k, l), (-h, -k, -l)]
    return np.array(indices)

def apply_trigonal_symmetry(hkls):
    """
    Apply trigonal symmetry to a list of Miller indices.
    
    Rules:
      - If h and k have the same sign, set both to their absolute values.
      - If h and k have different signs, adjust them so that h becomes negative and k becomes positive.
      - Set l to its absolute value.
      - Ensure |h| <= |k| by swapping if necessary.
    
    Parameters:
        hkls (array-like): List or array of Miller indices.
    
    Returns:
        np.ndarray: Array of symmetrized and unique Miller indices.
    """
    hkls = np.array(hkls)
    hkls[:, 2] = np.abs(hkls[:, 2])  # Ensure l is positive
    
    # Enforce |h| <= |k| by swapping where necessary.
    abs_h = np.abs(hkls[:, 0])
    abs_k = np.abs(hkls[:, 1])
    swap = abs_h > abs_k
    temp_h = hkls[swap, 0].copy()
    temp_k = hkls[swap, 1].copy()
    hkls[swap, 0] = temp_k
    hkls[swap, 1] = temp_h
    
    h = hkls[:, 0]
    k = hkls[:, 1]
    
    # Apply symmetry rules based on the signs of h and k.
    same_sign = (h * k) >= 0
    h[same_sign] = np.abs(h[same_sign])
    k[same_sign] = np.abs(k[same_sign])
    h[~same_sign] = np.abs(h[~same_sign])
    k[~same_sign] = -np.abs(k[~same_sign])
    # Adjust k for different signs (example rule; adjust as needed)
    i = -(h[~same_sign] + k[~same_sign])
    k[~same_sign] = i
    hkls[:, 0] = h
    hkls[:, 1] = k
    
    # Ensure |h| <= |k| again after adjustments.
    abs_h = np.abs(hkls[:, 0])
    abs_k = np.abs(hkls[:, 1])
    swap = abs_h > abs_k
    temp_h = hkls[swap, 0].copy()
    temp_k = hkls[swap, 1].copy()
    hkls[swap, 0] = temp_k
    hkls[swap, 1] = temp_h
    
    return np.array(hkls)

def apply_triclinic_symmetry(hkls):
    """
    Apply triclinic symmetry by ensuring the Miller indices are all non-negative.
    
    Parameters:
        hkls (array-like): List or array of Miller indices.
    
    Returns:
        np.ndarray: Modified Miller indices.
    """
    hkls = np.array(hkls)
    h_neg = (hkls[:, 0] < 0)
    hkls[h_neg] *= -1
    return hkls

def apply_monoclinic_symmetry(hkls):
    """
    Apply monoclinic symmetry to a list of Miller indices.
    
    For monoclinic symmetry:
      - Ensure the second index is positive.
      - Adjust h and l based on their signs.
    
    Parameters:
        hkls (array-like): List or array of Miller indices.
    
    Returns:
        np.ndarray: Array of symmetrized Miller indices.
    """
    hkls = np.array(hkls)
    hkls[:, 1] = np.abs(hkls[:, 1])
    
    h = hkls[:, 0]
    l = hkls[:, 2]
    same_sign = (h * l) >= 0
    hkls[same_sign] = np.abs(hkls[same_sign])
    h[~same_sign] = np.abs(h[~same_sign])
    l[~same_sign] = -np.abs(l[~same_sign])
    hkls[~same_sign, 0] = h[~same_sign]
    hkls[~same_sign, 2] = l[~same_sign]
    
    return hkls

def apply_tetragonal_symmetry(hkls):
    """
    Apply tetragonal symmetry to Miller indices.
    
    For tetragonal symmetry, take the absolute value and ensure the ordering of h and k.
    
    Parameters:
        hkls (array-like): List or array of Miller indices.
    
    Returns:
        np.ndarray: Array of symmetrized Miller indices.
    """
    hkls = np.array(hkls)
    hkls = np.abs(hkls)
    # For each row, sort h and k in descending order while keeping l unchanged.
    hkls = np.array([sorted(row[:2], reverse=True) + [row[2]] for row in hkls])
    return hkls

def apply_orthorhombic_symmetry(hkls):
    """
    Apply orthorhombic symmetry by taking the absolute values of Miller indices.
    
    Parameters:
        hkls (array-like): List or array of Miller indices.
    
    Returns:
        np.ndarray: Array of symmetrized Miller indices.
    """
    hkls = np.array(hkls)
    hkls = np.abs(hkls)
    return hkls

def apply_cubic_symmetry(hkls):
    """
    Apply cubic symmetry to Miller indices.
    
    For cubic symmetry, take the absolute values and sort each set of indices so that h >= k >= l.
    
    Parameters:
        hkls (array-like): List or array of Miller indices.
    
    Returns:
        np.ndarray: Array of symmetrized Miller indices.
    """
    hkls = np.array(hkls)
    hkls = np.abs(hkls)
    # Sort each row in descending order.
    hkls = np.array([sorted(row, reverse=True) for row in hkls])
    return hkls
