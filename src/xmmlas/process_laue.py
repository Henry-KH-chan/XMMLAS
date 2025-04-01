#!/usr/bin/env python3
"""
Process Laue Images and Generate Indexed Results
--------------------------------------------------
This script processes raw Laue images, performs indexing using a trained model,
and saves the results (including rotation/orientation matrices, indexed points, 
and HKL assignments) into an HDF5 file. The output is optimized for later use 
in generating scatter plots of the indexed Laue pattern, as well as for further 
analysis of the orientation matrix and other details.

Workflow:
1. Load the trained model data (weights, HKLs, and mask) and prepare file paths.
2. Identify the raw .tif Laue image files and filter out those already processed.
3. For each unprocessed file:
   - Load the Laue image using the provided mask.
   - Detect local maxima and convert pixel coordinates to (2θ, χ) values.
   - Initialize a LauePatternIndexer with the first 1000 detected points.
   - Perform multiple indexing attempts to obtain robust results.
   - Save each result in a separate group within an HDF5 file, storing:
       • rotation_matrix (orientation matrix for the indexed pattern)
       • indexed_points (coordinates for scatter plot generation)
       • indexed_hkls (assigned Miller indices for each spot)
       • closest_laue_points and closest_laue_hkls (for reference and further analysis)
4. Use multithreading to process multiple images concurrently.
"""

import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import necessary modules from the project
from utils.Laue_Image import LaueImage
from utils.indexer import LauePatternIndexer
from utils.symmetry import Symmetry
from utils.generate_hkl import simulate_laue_spots  # Currently unused; remove if unnecessary.
from utils.rotation_matrix import compute_rotation_matrix, convert_angle_to_direction
from utils.distance import compute_angular_distance

########################################
# Configuration
########################################

key_material = 'BaTiO3'
symmetry = Symmetry.cubic

# Input and output paths for raw and processed data respectively.
input_path = os.path.join("..", "..", "data", "raw", "BaTiO3")
output_path = os.path.join("..", "..", "data", "processed", "BaTiO3")

# Parameters for Laue image processing and indexing:
MIN_DISTANCE = 3         # Minimum distance for peak detection in find_local_maxima
THRESHOLD_MUL = 5        # Multiplier for thresholding in peak detection
TOLERANCE_PARAM = 0.15   # Tolerance parameter for LauePatternIndexer
MIN_NODES = 5            # Minimum nodes for identifying significant HKL pairs
INITIAL_NUM_PTS = 20     # Initial number of points for indexing process
INCREMENT = 10           # Increment step for the indexing process
EARLY_EXIT_NUM = 50      # Early exit parameter for indexing process
num_indexing_attempts = 10  # Number of indexing attempts to perform

# Ensure the output directory exists.
os.makedirs(output_path, exist_ok=True)

########################################
# Load Model Data
########################################

# Construct file paths for model weights and HKLs.
weight1_file = os.path.join('model', f'{key_material}_weight1.npy')
weight2_file = os.path.join('model', f'{key_material}_weight2.npy')
hkls_file    = os.path.join('model', f'{key_material}_hkls.npy')

# Load weights and HKLs. If saved as a tuple, allow_pickle=True might be necessary.
weight1 = np.load(weight1_file)
weight2 = np.load(weight2_file)
weights = (weight1, weight2)
hkls = np.load(hkls_file)

# Load the mask file (assumed to be used for masking the raw Laue image).
mask = np.load('mask/pilatus.npy')

########################################
# File Processing Setup
########################################

# List all .tif files in the input directory.
files = [file for file in os.listdir(input_path) if file.endswith('.tif')]

# Determine which files have already been processed by checking output filenames.
# The processed files have filenames ending with '_results.h5'. Adjust this logic if needed.
processed_files = [os.path.splitext(file)[0].replace('_results', '') + '.tif'
                   for file in os.listdir(output_path) if file.endswith('_results.h5')]
unprocessed_files = [file for file in files if file not in processed_files]

print(f"Total files: {len(files)}")
print(f"Processed files: {len(processed_files)}")
print(f"Unprocessed files: {len(unprocessed_files)}")

########################################
# Processing Function
########################################

def process_file(file):
    """
    Process a single Laue image file: detect peaks, perform indexing, and save the results.
    
    The results are stored in an HDF5 file (one per input image) with groups for each indexing attempt.
    Each group contains datasets for the rotation matrix (orientation), indexed points for scatter plot,
    indexed HKLs, and the closest Laue points and HKLs.
    
    Args:
        file (str): Filename of the Laue image.
    
    Returns:
        tuple: (file, success flag) indicating the processing result.
    """
    try:
        import h5py  # Import h5py within the worker to ensure thread safety.
        print(f"Processing file: {file}")
        
        # Load the Laue image with the provided mask.
        laue_image = LaueImage(os.path.join(input_path, file), mask)
        
        # Find local maxima and convert pixel coordinates to (2θ, χ) angles.
        laue_image.find_local_maxima(min_distance=MIN_DISTANCE, threshold_mul=THRESHOLD_MUL)
        laue_image.convert_pixels_to_theta_chi()

        # Extract (2θ, χ) points and intensity values.
        twotheta_chi_points = laue_image.twotheta_chi_points
        intensity = laue_image.intensity

        # Initialize the LauePatternIndexer with a subset of points for efficiency.
        indexer = LauePatternIndexer(
            twotheta_chi_points,
            intensity,
            weights,
            hkls,
            key_material,
            symmetry,
            TOLERANCE_PARAM  # Tolerance parameter; adjust if necessary.
        )

        # Perform multiple indexing attempts to improve robustness.
        results = []
        for i in range(num_indexing_attempts):
            indexer.identify_significant_hkl_pairs(min_nodes=MIN_NODES, initial_num_pts=INITIAL_NUM_PTS, increment=INCREMENT)
            result = indexer.indexing(early_exit_num=EARLY_EXIT_NUM)
            if result is not None:
                results.append(result)

        # Define a unique filename for the output results (HDF5 format).
        result_filename = os.path.splitext(file)[0] + "_results.h5"
        
        # Save the results if available.
        if results:
            with h5py.File(os.path.join(output_path, result_filename), 'w') as f:
                # For each successful indexing result, create a group in the HDF5 file.
                for idx, res in enumerate(results):
                    grp = f.create_group(f'result_{idx}')
                    # Save the orientation matrix (rotation_matrix) for scatter plot and further analysis.
                    grp.create_dataset('rotation_matrix', data=res[0])
                    # Save indexed points which can be used to generate a scatter plot.
                    grp.create_dataset('indexed_points', data=res[1])
                    # Save the corresponding HKL assignments.
                    grp.create_dataset('indexed_hkls', data=res[2])
                    # Save additional details for reference (e.g., closest Laue pattern points and HKLs).
                    grp.create_dataset('closest_laue_points', data=res[3])
                    grp.create_dataset('closest_laue_hkls', data=res[4])
                    # Optionally, store attributes (e.g., processing parameters or metadata)
                    grp.attrs['description'] = "Orientation matrix, indexed points, HKLs, and reference details"
            print(f"Finished processing file: {file}")
        else:
            # If no results were obtained, save an empty file with an attribute noting no results.
            with h5py.File(os.path.join(output_path, result_filename), 'w') as f:
                f.attrs['no_results'] = True
            print(f"No indexing results for file: {file}")

        return file, True  # Indicate success

    except Exception as e:
        print(f"Error processing file {file}: {e}")
        return file, False  # Indicate failure

########################################
# Main Execution with Multithreading
########################################

if __name__ == '__main__':
    # Use ThreadPoolExecutor to process multiple files concurrently.
    with ThreadPoolExecutor() as executor:
        # Submit tasks for all unprocessed files.
        futures = {executor.submit(process_file, file): file for file in unprocessed_files}

        # Collect results as they complete.
        for future in as_completed(futures):
            file = futures[future]
            try:
                file_processed, success = future.result()
                if success:
                    print(f"Successfully processed {file_processed}")
                else:
                    print(f"Failed to process {file_processed}")
            except Exception as e:
                print(f"Exception occurred while processing {file}: {e}")
