import numpy as np
import random
from typing import List, Tuple

try:
    # Try relative import first
    from .Laue_pattern import LauePattern
except ImportError:
    # Fall back to absolute import if the relative import fails
    from Laue_pattern import LauePattern


def generate_overlap_laue_patterns(laue_patterns: List[LauePattern], num_to_combine: int) -> LauePattern:
    """
    Randomly selects 'num_to_combine' Laue patterns from the list, stacks their 
    filtered points and hkls, and returns a new LauePattern with combined data.

    Parameters:
        laue_patterns (List[LauePattern]): List of LauePattern objects.
        num_to_combine (int): Number of patterns to combine.

    Returns:
        LauePattern: A new LauePattern with stacked points and hkls.
    """
    if num_to_combine > len(laue_patterns):
        raise ValueError("num_to_combine exceeds available number of Laue patterns.")

    selected_patterns = random.sample(laue_patterns, num_to_combine)
    # Collect hkls and points from each selected pattern
    hkls_list = [pattern.filtered_hkls for pattern in selected_patterns]
    points_list = [pattern.filtered_points for pattern in selected_patterns]
    
    # Combine points and hkls using vertical stacking
    combined_points = np.vstack(points_list)
    combined_hkls = np.vstack(hkls_list)
    # Return a new LauePattern with the combined data
    return LauePattern(combined_points, combined_hkls)


def generate_overlap_hist_hkls(laue_patterns: List[LauePattern], nb_patterns: int = 100, 
                               max_hkl: int = 6, max_overlapped: int = 5, 
                               data_augmentation: bool = True) -> Tuple[np.ndarray, List[Tuple], int]:
    """
    Generate overlapped histograms and hkls by combining multiple Laue patterns.
    It generates several combined Laue patterns by varying the number of overlapped patterns,
    then computes histograms and hkls from these combined patterns.

    Parameters:
        laue_patterns (List[LauePattern]): List of original LauePattern objects.
        nb_patterns (int): Number of random combinations per overlap count.
        max_hkl (int): Maximum HKL value to be considered.
        max_overlapped (int): Maximum number of patterns to overlap.
        data_augmentation (bool): Whether to apply data augmentation to each pattern.

    Returns:
        Tuple[np.ndarray, List[Tuple], int]:
            - A NumPy array of stacked and normalized histograms.
            - A list of HKL tuples.
            - Wavelengths (currently set to 0).
    """
    # Generate combined patterns: for each overlap count from 1 to max_overlapped,
    # repeat nb_patterns times.
    combined_patterns = [
        generate_overlap_laue_patterns(laue_patterns, i + 1) 
        for i in range(max_overlapped) 
        for _ in range(nb_patterns)
    ]
    
    # Generate histograms, hkls, and wavelengths from the combined patterns.
    histograms, hkls, wavelengths = generate_hist_hkls(combined_patterns, max_hkl, data_augmentation)
    
    # Stack histograms vertically
    histograms = np.vstack(histograms)
    # Create a boolean mask to filter out nearly empty histograms
    valid_mask = np.sum(histograms, axis=1) > 1
    histograms = histograms[valid_mask]
    
    hkls_stacked = np.vstack(hkls)[valid_mask]
    # Convert hkls array to a list of tuples
    hkls_list = [tuple(hkl) for hkl in hkls_stacked]
    
    # Normalize each histogram row by its maximum value (avoid division by zero)
    max_vals = np.max(histograms, axis=1)
    # A small epsilon can be added if needed for safety, e.g., max_vals + 1e-8
    histograms = histograms / max_vals[:, np.newaxis]
    
    return histograms, hkls_list, wavelengths


def generate_hist_hkls(laue_patterns: List[LauePattern], max_hkl: int, 
                       data_augmentation: bool = True) -> Tuple[List[np.ndarray], List[np.ndarray], int]:
    """
    Generate histograms, hkls, and wavelengths from a list of Laue patterns.
    Optionally applies data augmentation, filters the points based on max_hkl,
    and computes histograms of angular distances.

    Parameters:
        laue_patterns (List[LauePattern]): List of LauePattern objects.
        max_hkl (int): Maximum HKL value for filtering.
        data_augmentation (bool): Whether to apply random data augmentation.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray], int]:
            - A list of histogram arrays.
            - A list of filtered HKL arrays.
            - Wavelengths (currently returned as 0).
    """
    histograms = []
    hkls = []
    wavelengths = 0  # Placeholder for wavelength information

    for pattern in laue_patterns:
        if data_augmentation:
            pattern.apply_random_augmentation()
        pattern.filter_points_based_on_hkl(max_hkl)
        pattern.calculate_histograms_of_distances()
        # Uncomment the following line if wavelength calculation is implemented:
        # pattern.calculate_wavelengths()
        if len(pattern.histograms) == 0:
            print("Warning: max_hkl is too small for this pattern; skipping.")
            continue
        
        histograms.append(pattern.histograms)
        hkls.append(pattern.filtered_hkls)

    return histograms, [np.array(h) for h in hkls], wavelengths
