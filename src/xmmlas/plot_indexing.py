#!/usr/bin/env python3
"""
This script reads a result HDF5 file produced by process_laue.py and plots
a scatter plot comparing experimental points and indexed (simulated) points.
It assumes each group in the HDF5 file contains at least the following datasets:
  - 'indexed_points'  : Experimental points (e.g., best_points)
  - 'closest_laue_points' : Indexed (or simulated) points corresponding to the experimental points.
  
Usage:
    python plot_indexing.py <result_file.h5> [output_image.png]
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys

def plot_indexing_results(h5_file_path, output_image='scatter_plot.png'):
    """
    Reads an HDF5 result file and creates a scatter plot comparing the experimental
    points (indexed_points) and the indexed (simulated) points (closest_laue_points).
    
    Parameters:
        h5_file_path (str): Path to the result HDF5 file.
        output_image (str): Filename for saving the output plot.
    """
    experimental_points_all = []
    indexed_points_all = []
    
    # Open the HDF5 file.
    with h5py.File(h5_file_path, 'r') as f:
        # Iterate over all groups (e.g., result_0, result_1, etc.)
        for group_name in f.keys():
            group = f[group_name]
            if 'indexed_points' in group and 'closest_laue_points' in group:
                exp_points = np.array(group['indexed_points'])
                idx_points = np.array(group['closest_laue_points'])
                experimental_points_all.append(exp_points)
                indexed_points_all.append(idx_points)
            else:
                print(f"Group '{group_name}' does not contain expected datasets; skipping.")

    if not experimental_points_all or not indexed_points_all:
        print("No valid groups with required datasets were found in the file.")
        return

    # Concatenate data from all groups.
    experimental_points = np.vstack(experimental_points_all)
    indexed_points = np.vstack(indexed_points_all)
    
    # Create scatter plot.
    plt.figure(figsize=(8, 6))
    plt.scatter(experimental_points[:, 0], experimental_points[:, 1],
                c='blue', label='Experimental Points', alpha=0.7)
    plt.scatter(indexed_points[:, 0], indexed_points[:, 1],
                c='red', marker='x', label='Indexed Points', alpha=0.7)
    plt.title('Experimental Points vs. Indexed Points')
    plt.xlabel('X Coordinate (pixels)')
    plt.ylabel('Y Coordinate (pixels)')
    plt.legend()
    plt.grid(True)
    
    # Save and display the plot.
    plt.savefig(output_image, dpi=300)
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python plot_indexing.py <result_file.h5> [output_image.png]")
        sys.exit(1)
    
    result_file = sys.argv[1]
    output_image = sys.argv[2] if len(sys.argv) > 2 else 'scatter_plot.png'
    plot_indexing_results(result_file, output_image)
