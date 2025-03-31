# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:28:49 2024

@author: KHChan
"""

import numpy as np
import random
from math import gcd
from functools import reduce

try:
    from .distance import compute_angular_distance  # renamed from angular_distance
    from .symmetry import apply_symmetry, Symmetry
except ImportError:
    from distance import compute_angular_distance  # renamed from angular_distance
    from symmetry import apply_symmetry, Symmetry

class LauePattern:
    def __init__(self, measurement_points,hkls, symmetry_group=Symmetry.tetragonal, lattice_parameters = (3.992, 3.992, 4.036, 90.0, 90.0, 90.0)):
        self.measurement_points = np.array(measurement_points)
        self.hkls = np.array(hkls)
        self.quaternions = []  # Unused, consider removing if not needed later
        self.symmetry_group = symmetry_group
        self.filtered_points = np.copy(self.measurement_points)
        self.filtered_hkls = np.copy(self.hkls)
        self.histograms = []
        
     
    def filter_points_based_on_hkl(self, hkl_limit=6):
        """
        Filters the hkls and corresponding measurement points based on hkl limits.
        Args:
        hkl_limit (int): The maximum absolute value for h, k, l.
        """
        hkl_mask = np.all(np.abs(self.filtered_hkls) <= hkl_limit, axis=1)
        self.filtered_points = self.filtered_points[hkl_mask]
        self.filtered_hkls = self.filtered_hkls[hkl_mask]
        
    def add_random_points(self):
        """
        Adds random points to the measurement points, with hkls set to (64, 64, 64).
        The number of points added depends on the number of points in the Laue pattern.
        """
        num_existing_points = self.measurement_points.shape[0]
        if num_existing_points > 0:
            
            num_points_to_add = num_existing_points  # You can adjust this ratio as needed
    
            # Generate random points within the range of existing measurement points
            min_values = self.measurement_points.min(axis=0)
            max_values = self.measurement_points.max(axis=0)
            random_points = np.random.uniform(low=min_values, high=max_values, size=(num_points_to_add, 2))
    
            # Set the corresponding hkls to (64, 64, 64)
            random_hkls = np.tile([-1, -1, -1], (num_points_to_add, 1))
    
            # Append the random points and hkls to the filtered arrays
            self.measurement_points = np.vstack((self.measurement_points, random_points))
            self.hkls = np.vstack((self.hkls, random_hkls))

    def add_noise_to_points(self, noise_level=0.1, noise_fraction=0.2):
        """
        Adds Gaussian noise to a fraction of the points.
        Args:
        noise_level (float): Standard deviation of the Gaussian noise.
        noise_fraction (float): Fraction of points to modify.
        """
        indices = np.random.choice(len(self.measurement_points), int(len(self.measurement_points) * noise_fraction), replace=False)
        noise = np.random.normal(0, noise_level, (len(indices), 2))
        self.measurement_points[indices] += noise

    def remove_random_peaks(self, remove_fraction=0.2):
        """
        Randomly removes a fraction of peaks from filtered points and hkls.
        Args:
        remove_fraction (float): Fraction of peaks to remove.
        """
        num_peaks = len(self.measurement_points)
        num_to_remove = int(num_peaks * remove_fraction)
        indices_to_remove = np.random.choice(num_peaks, num_to_remove, replace=False)
        self.measurement_points = np.delete(self.measurement_points, indices_to_remove, axis=0)
        # self.filtered_hkls = np.delete(self.filtered_hkls, indices_to_remove, axis=0)

    def apply_random_augmentation(self):
        """
        Applies a random augmentation (noise addition, peak removal, adding random points, or none) to the data.
        """
        # self.add_random_points()
        # self.remove_random_peaks(remove_fraction=0.75)
        action = random.choice(['noise', 'remove', 'both', 'none'])
        if action == 'noise':
            self.add_noise_to_points(noise_level=0.1, noise_fraction=0.2)
        elif action == 'remove':
            self.remove_random_peaks(remove_fraction=0.5)
        elif action == 'both':
            self.add_noise_to_points(noise_level=0.05, noise_fraction=0.1)
            self.remove_random_peaks(remove_fraction=0.25)

    def calculate_histograms_of_distances(self):
        """
        Calculates histograms of angular distances between filtered points and all points.
        """
        if not len(self.filtered_points):
            return
        #self.filtered_points = np.vstack((self.filtered_points,(90,0)))
        distances = compute_angular_distance(self.filtered_points, self.measurement_points)
        distances[np.abs(distances) < 0.1] = np.inf  # Replace small distances to avoid self-comparison issues
        self.histograms = np.array([np.histogram(dist, bins=900, range=(0, 90))[0] for dist in distances])
        self.histograms = self.histograms

    def reduce_hkl(self, hkl):
        """
        Reduces an hkl tuple to its simplest form.
        Args:
        hkl (tuple): The hkl values to be reduced.
        Returns:
        tuple: The reduced hkl values.
        """
        common_factor = reduce(gcd, hkl)
        return tuple(int(value / common_factor) for value in hkl)

    def synchronize_hkls_with_points(self):
        """
        Reduces all hkls to their simplest form and synchronizes with corresponding points to remove duplicates.
        """
        reduced_hkls = np.array([self.reduce_hkl(tuple(hkl)) for hkl in self.filtered_hkls])
        unique_hkls, indices = np.unique(reduced_hkls, axis=0, return_index=True)
        self.filtered_hkls = unique_hkls
        self.filtered_points = self.filtered_points[indices]

    def apply_symmetry_operations(self):
        """
        Applies symmetry operations to hkls based on the specified symmetry group.
        """
        self.filtered_hkls = apply_symmetry(self.filtered_hkls, self.symmetry_group)
        
    def calculate_wavelengths(self):
        self.calculate_d_spacing_for_all_hkls()
        self.wavelengths = 2*self.dspacing*np.sin(np.radians(self.filtered_points[:,0])/2)