# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 10:11:01 2024

@author: KHChan
"""

import sys
sys.path.append("..")
import numpy as np
from LaueTools.CrystalParameters import Prepare_Grain
from LaueTools.dict_LaueTools import dict_Materials
import networkx as nx

# Attempt to handle both relative and absolute imports depending on the environment
try:
    from .distance import compute_angular_distance, calculate_angles_between_hkls
    from .generate_hkl import simulate_laue_spots
    from .rotation_matrix import compute_rotation_matrix, convert_direction_to_angle
    from .optimizer import optimize_laue_points
    from .optimize_R import optimize_orientation
    from .symmetry import generate_symmetric_indices, Symmetry
except ImportError:
    from distance import compute_angular_distance, calculate_angles_between_hkls
    from generate_hkl import simulate_laue_spots
    from rotation_matrix import compute_rotation_matrix, convert_direction_to_angle
    from optimizer import optimize_laue_points
    from optimize_R import optimize_orientation
    from symmetry import generate_symmetric_indices, Symmetry


class LauePatternIndexer:
    """
    A class to index Laue patterns using detected points, model weights, and HKL indices.
    It identifies significant cliques of HKL pairs and optimizes rotation matrices for indexing.
    """

    def __init__(self, laue_points, intensity, model_weight, hkls, material_names, symmetry=Symmetry.cubic, angular_tolerance=0.5):
        """
        Initialize the LauePatternIndexer with measurement points, intensities, model weights, and material properties.

        Args:
            laue_points (np.ndarray): Array of detected Laue points.
            intensity (np.ndarray): Array of intensities corresponding to Laue points.
            model_weight (tuple): Tuple of model weights (weights[0], weights[1]).
            hkls (np.ndarray): Array of HKL indices.
            material_names (list): List of material names.
            symmetry (Symmetry): Symmetry of the crystal structure.
            angular_tolerance (float): Tolerance in angular differences for matching.
        """
        self.initialize_data(laue_points, intensity)
        self.material_names = material_names
        self.symmetry = symmetry
        self.angular_tolerance = angular_tolerance
        self.hkls = hkls

        # Compute angular distances between points
        self.angular_distances = compute_angular_distance(self.laue_points, self.laue_points)
        # Replace very small distances (likely self-comparisons) with infinity to avoid false matches
        self.angular_distances[np.abs(self.angular_distances) < 0.1] = np.inf

        self.weights = model_weight
        self.prepare_grain()
        
        # Flag to indicate if no cliques can be found during indexing
        self.no_clique_found = False

    def initialize_data(self, laue_points, intensity):
        """
        Sort Laue points and intensities in descending order based on intensity.

        Args:
            laue_points (np.ndarray): Array of detected Laue points.
            intensity (np.ndarray): Array of intensities corresponding to Laue points.
        """
        sorted_indices = np.argsort(intensity)[::-1]
        self.laue_points = laue_points[sorted_indices]
        self.intensity = intensity[sorted_indices]

    def prepare_grain(self):
        """
        Prepare the grain based on material names and an orientation matrix.
        Uses an identity matrix as a default orientation.
        """
        orientation_matrix = np.eye(3)  # Default orientation
        grain = Prepare_Grain(self.material_names, orientation_matrix, dictmaterials=dict_Materials)
        self.B_matrix = grain[0]  # Assuming Prepare_Grain returns a tuple/list; index 0 is B_matrix

    def compute_histograms(self, num_pts=30):
        """
        Compute histograms of angular distances between Laue points.

        Args:
            num_pts (int): Number of top Laue points to use for histogram computation.
        """
        distances = self.angular_distances[:num_pts, :num_pts]
        histograms = np.array([
            np.histogram(dist, bins=900, range=(0, 90))[0]
            for dist in distances
        ])
        # Normalize each histogram row; avoid division by zero.
        hist_max = np.max(histograms, axis=1)
        hist_max[hist_max == 0] = 1
        self.histograms = histograms / hist_max[:, np.newaxis]

    def apply_model_weights(self, num_pts=30):
        """
        Apply model weights to histograms and extract HKL indices.

        Args:
            num_pts (int): Number of top Laue points to consider.
        """
        # Compute weighted scores from histograms using a simple linear model.
        weighted_scores = self.histograms @ self.weights[0] + self.weights[1]
        self.hkls_ind = np.argmax(weighted_scores, axis=1)
        self.valid_hkls = self.hkls[self.hkls_ind]

        # Transform HKLs using the B_matrix
        transformed_hkls = (self.B_matrix @ self.valid_hkls.T).T
        self.predicted_angles = calculate_angles_between_hkls(transformed_hkls)

        # Compute angular differences between experimental points and predicted angles.
        point_distances = self.angular_distances[:num_pts, :num_pts]
        self.angle_diff = np.abs(point_distances - self.predicted_angles)
        self.sum_angle_diff = np.sum(self.angle_diff < (self.angular_tolerance / 2), axis=0)

    def identify_significant_hkl_pairs(self, min_nodes=5, initial_num_pts=30, increment=10):
        """
        Identify significant cliques of HKL pairs based on angular tolerance.
        Iteratively increases the number of points used until significant cliques are found
        or all points have been tried.

        Args:
            min_nodes (int): Minimum number of nodes required in a clique to be considered significant.
            initial_num_pts (int): Initial number of top Laue points to use.
            increment (int): Increment step to increase points if no significant cliques are found.
        """
        self.significant_cliques = None
        if self.no_clique_found:
            print("No cliques can be found. Skipping identification.")
            return

        total_points = self.laue_points.shape[0]
        num_pts = initial_num_pts

        while num_pts <= total_points:
            self.compute_histograms(num_pts=num_pts)
            self.apply_model_weights(num_pts=num_pts)

            # Create a graph where each node represents an HKL from the top 'num_pts' points.
            graph = nx.Graph()
            for i, hkl in enumerate(self.valid_hkls[:num_pts]):
                graph.add_node(i, hkl=tuple(hkl), point=tuple(self.laue_points[i]))

            # Add edges based on angular tolerance and a minimum matching condition.
            for i in range(len(self.valid_hkls[:num_pts])):
                for j in range(i + 1, len(self.valid_hkls[:num_pts])):
                    if (
                        not np.array_equal(self.valid_hkls[i], self.valid_hkls[j]) and
                        self.angle_diff[i, j] < (self.angular_tolerance / 2) and
                        np.all(self.sum_angle_diff[[i, j]] > 4)
                    ):
                        graph.add_edge(i, j)

            self.graph = graph

            try:
                cliques = list(nx.find_cliques(graph))
            except Exception as e:
                print(f"Failed to compute cliques due to: {str(e)}")
                return

            if not cliques:
                if num_pts < total_points:
                    new_num_pts = min(num_pts + 5, total_points)
                    num_pts = new_num_pts
                    self.identify_significant_hkl_pairs(min_nodes=min_nodes, initial_num_pts=num_pts)
                    continue
                else:
                    print("No cliques found even after using all available points.")
                    self.no_clique_found = True
                    return

            # Filter cliques for those with at least 'min_nodes' nodes.
            large_cliques = [clique for clique in cliques if len(clique) >= min_nodes]
            if not large_cliques:
                if num_pts < total_points:
                    new_num_pts = min(num_pts + increment, total_points)
                    num_pts = new_num_pts
                    continue
                else:
                    self.no_clique_found = True
                    return

            try:
                # Choose the clique with the maximum sum of intensities.
                clique_intensities = [np.sum(self.intensity[clique]) for clique in large_cliques]
                large_clique_ind = np.argmax(clique_intensities)
                clique = large_cliques[large_clique_ind]
                self.significant_cliques = list(clique)
                return  # Successful identification
            except Exception as e:
                print(f"Error calculating or assigning the significant cliques: {str(e)}")
                return

        if not self.significant_cliques:
            print("Failed to identify any significant cliques after all checks.")
            self.no_clique_found = True

    def indexing(self, early_exit_num):
        """
        Perform indexing based on laue_points and HKL values, excluding outliers based on angular difference.

        Returns:
            tuple: (optimized_R_matrix, best_points, best_hkls, new_points, new_hkls, best_intensities)
        """
        if not self.significant_cliques:
            print('No significant cliques for indexing')
            return None

        # Extract grouped data from significant cliques.
        grouped_points = self.laue_points[self.significant_cliques]
        grouped_hkls = self.valid_hkls[self.significant_cliques]

        # Compute indices of the top two HKLs based on a sorting criterion.
        top_hkl_indices = self._get_top_two_hkl_indices(grouped_hkls)
        new_hkl0 = generate_symmetric_indices(grouped_hkls[top_hkl_indices[0]], symmetry=self.symmetry)
        new_hkl1 = generate_symmetric_indices(grouped_hkls[top_hkl_indices[1]], symmetry=self.symmetry)

        # Find the best rotation matrix by iterating over possible combinations.
        best_R = self._find_best_rotation_matrix(grouped_points, top_hkl_indices, new_hkl0, new_hkl1, early_exit_num=200)
        if best_R is None:
            return None

        # First round: simulate and match to obtain initial matches.
        matched_indices, best_hkls = self._simulate_and_match(self.laue_points, self.intensity, best_R)
        if len(matched_indices) < 8:
            return None

        # Optimize the rotation matrix using the initial matches.
        best_points = self.laue_points[matched_indices]
        self.optimized_R_matrix = optimize_orientation(best_points, self.B_matrix, best_hkls)[0]

        # Second round: refine simulation and matching.
        matched_indices, best_hkls, new_points, new_hkls = self._second_round_simulation_and_matching(
            self.laue_points, self.intensity, self.optimized_R_matrix
        )
        if len(matched_indices) < 8:
            return None

        best_points = self.laue_points[matched_indices]
        best_intensities = self.intensity[matched_indices]
        self.optimized_R_matrix = optimize_orientation(best_points, self.B_matrix, best_hkls)[0]

        # Remove matched points from the dataset.
        valid_mask = np.ones(len(self.laue_points), dtype=bool)
        valid_mask[matched_indices] = False
        self.laue_points = self.laue_points[valid_mask]
        self.intensity = self.intensity[valid_mask]
        self.angular_distances = self.angular_distances[np.ix_(valid_mask, valid_mask)]

        return self.optimized_R_matrix, best_points, best_hkls, new_points, new_hkls, best_intensities

    def _get_top_two_hkl_indices(self, grouped_hkls):
        """
        Returns indices of the top two HKLs based on the smallest sum of absolute HKL values.
        
        Args:
            grouped_hkls (np.ndarray): Array of grouped HKLs.
        
        Returns:
            np.ndarray: Indices of the two best HKL candidates.
        """
        return np.argsort(np.sum(np.abs(grouped_hkls), axis=1))[:2]

    def _find_best_rotation_matrix(self, grouped_points, top_indices, new_hkl0, new_hkl1, early_exit_num=200):
        """
        Iterate over combinations of symmetric HKLs to find the best rotation matrix.
        
        Args:
            grouped_points (np.ndarray): Points corresponding to significant cliques.
            top_indices (np.ndarray): Indices of the top two HKLs.
            new_hkl0, new_hkl1: Arrays of symmetric HKL candidates.
            early_exit_num (int): Early exit threshold for matched points.
        
        Returns:
            np.array or None: Best rotation matrix or None if not found.
        """
        best_R = None
        max_match_points = -1
        ref_points = grouped_points[top_indices]

        for hkl0 in new_hkl0:
            for hkl1 in new_hkl1:
                # Skip if hkl0 and hkl1 are identical or negatives.
                if np.all(hkl0 == hkl1) or np.all(hkl0 == -hkl1):
                    continue
                R = compute_rotation_matrix(ref_points, [hkl0, hkl1], self.B_matrix)
                simulated_points = simulate_laue_spots(R, self.material_names)[0]

                match_points = self._count_matches(self.laue_points, simulated_points, 2 * self.angular_tolerance)

                if match_points > max_match_points:
                    max_match_points = match_points
                    best_R = R
                    if match_points > early_exit_num:
                        print(hkl0)
                        print(hkl1)
                        print(f"Early exit: {match_points} matches found with current rotation matrix.")
                        return best_R
        return best_R

    def _count_matches(self, laue_points, simulated_points, tolerance):
        """
        Counts the number of matches between laue_points and simulated_points within a tolerance.
        
        Args:
            laue_points (np.ndarray): Experimental Laue points.
            simulated_points (np.ndarray): Simulated Laue points.
            tolerance (float): Angular tolerance for matching.
        
        Returns:
            int: Number of matched points.
        """
        differences = np.linalg.norm(laue_points[:, np.newaxis, :] - simulated_points[np.newaxis, :, :], axis=2)
        return np.sum(differences < tolerance)

    def _simulate_and_match(self, laue_points, intensity, R_matrix):
        """
        Simulate Laue spots from a rotation matrix and match them with experimental data.
        
        Args:
            laue_points (np.ndarray): Experimental Laue points.
            intensity (np.ndarray): Corresponding intensities.
            R_matrix (np.array): Rotation matrix for simulation.
        
        Returns:
            tuple: (matched_indices, matched_hkls)
        """
        new_points, new_hkls = simulate_laue_spots(R_matrix, self.material_names)
        angular_distances = compute_angular_distance(laue_points, new_points)
        below_threshold = angular_distances < self.angular_tolerance

        matched_indices, matched_hkls = self._get_best_matches(laue_points, intensity, new_points, new_hkls, below_threshold)
        return matched_indices, matched_hkls

    def _second_round_simulation_and_matching(self, laue_points, intensity, R_matrix):
        """
        Perform a second round of simulation and matching after an initial optimization.
        
        Args:
            laue_points (np.ndarray): Experimental Laue points.
            intensity (np.ndarray): Corresponding intensities.
            R_matrix (np.array): Optimized rotation matrix.
        
        Returns:
            tuple: (matched_indices, matched_hkls, new_points, new_hkls)
        """
        new_points, new_hkls = simulate_laue_spots(R_matrix, self.material_names)

        # Filter simulated points based on camera limits (hard-coded thresholds).
        points_filter = np.all([
            new_points[:, 0] < 105,
            new_points[:, 0] > 49,
            new_points[:, 1] > -31,
            new_points[:, 1] < 31
        ], axis=0)

        new_points = new_points[points_filter]
        new_hkls = new_hkls[points_filter]

        angular_distances = compute_angular_distance(laue_points, new_points)
        below_threshold = angular_distances < self.angular_tolerance

        matched_indices, matched_hkls = self._get_best_matches(laue_points, intensity, new_points, new_hkls, below_threshold)
        return matched_indices, matched_hkls, new_points, new_hkls

    def _get_best_matches(self, laue_points, intensity, sim_points, sim_hkls, below_threshold):
        """
        For each simulated point, finds the best matching experimental point within a threshold.
        The match with the highest intensity is selected.
        
        Args:
            laue_points (np.ndarray): Experimental Laue points.
            intensity (np.ndarray): Intensities for the experimental points.
            sim_points (np.ndarray): Simulated points.
            sim_hkls (np.ndarray): Simulated HKLs corresponding to sim_points.
            below_threshold (np.array): Boolean matrix indicating which points are within tolerance.
        
        Returns:
            tuple: (matched_indices, matched_hkls) as arrays.
        """
        matched_indices = []
        best_hkls = []

        for i, (sp, shkl) in enumerate(zip(sim_points, sim_hkls)):
            valid_within_threshold = below_threshold[:, i]
            if np.any(valid_within_threshold):
                intensities_within_threshold = intensity[valid_within_threshold]
                max_intensity_index = np.argmax(intensities_within_threshold)
                best_match_index = np.where(valid_within_threshold)[0][max_intensity_index]
                matched_indices.append(best_match_index)
                best_hkls.append(shkl)

        return np.array(matched_indices), np.array(best_hkls)
