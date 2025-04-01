import numpy as np

# Constant for converting degrees to radians.
DEGREES_TO_RADIANS = np.pi / 180  # Conversion factor from degrees to radians

def compute_angular_distance(points1, points2):
    """
    Computes angular distance between two sets of points given in polar coordinates (theta, chi).
    Each point is specified in degrees and the function returns distances in degrees.
    
    Args:
        points1 (np.array): Array of points with shape (N, 2) where each row is (theta, chi).
        points2 (np.array): Array of points with shape (M, 2) where each row is (theta, chi).
    
    Returns:
        np.array: Matrix of angular distances in degrees with shape (N, M).
    """
    # Convert theta angles from degrees to radians, and divide by 2.
    theta1 = points1[:, 0] * DEGREES_TO_RADIANS / 2
    full_chi1 = points1[:, 1] * DEGREES_TO_RADIANS

    theta2 = points2[:, 0] * DEGREES_TO_RADIANS / 2
    full_chi2 = points2[:, 1] * DEGREES_TO_RADIANS

    # Compute differences in chi for each pair: shape becomes (N, M)
    chi_differences = full_chi1[:, np.newaxis] - full_chi2
    # Compute the cosine of the angular distance using spherical trigonometry
    cos_angle = (np.sin(theta1[:, np.newaxis]) * np.sin(theta2) +
                 np.cos(theta1[:, np.newaxis]) * np.cos(theta2) * np.cos(chi_differences))
    # Clip to ensure the values remain valid for arccos
    cos_angle = np.clip(cos_angle, -1, 1)

    angular_distances = np.arccos(cos_angle)
    return angular_distances / DEGREES_TO_RADIANS  # Convert radians back to degrees

def compute_angular_distance2(points1, points2):
    """
    Computes angular distance between two sets of points given in polar coordinates (theta, chi).
    Each point is specified in degrees and the function returns distances in degrees.
    
    This variant computes the difference in chi without using broadcasting for chi_differences.
    
    Args:
        points1 (np.array): Array of points with shape (N, 2) where each row is (theta, chi).
        points2 (np.array): Array of points with shape (M, 2) where each row is (theta, chi).
    
    Returns:
        np.array: Matrix of angular distances in degrees with shape (N, M).
    """
    theta1 = points1[:, 0] * DEGREES_TO_RADIANS / 2
    full_chi1 = points1[:, 1] * DEGREES_TO_RADIANS

    theta2 = points2[:, 0] * DEGREES_TO_RADIANS / 2
    full_chi2 = points2[:, 1] * DEGREES_TO_RADIANS

    # Here, difference is computed without expanding dimensions
    chi_differences = full_chi1 - full_chi2
    # Compute cosine using vectorized operations; note that this approach 
    # assumes points1 and points2 are 1-D arrays for chi, which may yield a different shape.
    cos_angle = (np.sin(theta1) * np.sin(theta2) +
                 np.cos(theta1) * np.cos(theta2) * np.cos(chi_differences))
    cos_angle = np.clip(cos_angle, -1, 1)

    angular_distances = np.arccos(cos_angle)
    return angular_distances / DEGREES_TO_RADIANS  # Convert radians back to degrees

def normalize_vectors(vectors):
    """
    Normalizes a list of vectors.
    
    Args:
        vectors (np.array): Array of vectors to be normalized.
    
    Returns:
        np.array: Normalized vectors.
    
    Raises:
        ValueError: If one or more vectors have zero magnitude.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("One or more vectors have zero magnitude, cannot normalize.")
    normalized_vectors = vectors / norms
    return normalized_vectors

def calculate_angles_between_hkls(hkls):
    """
    Calculates the angular distance between each pair of HKL vectors using the dot product.
    
    Args:
        hkls (np.array): Array of HKL vectors.
    
    Returns:
        np.array: Angular distances in degrees between each pair of vectors.
    """
    normalized_vectors = normalize_vectors(hkls)
    dot_products = np.dot(normalized_vectors, normalized_vectors.T)
    dot_products = np.clip(dot_products, -1.0, 1.0)  # Clip values for arccos validity
    angles_in_radians = np.arccos(dot_products)
    return angles_in_radians / DEGREES_TO_RADIANS  # Convert radians to degrees
