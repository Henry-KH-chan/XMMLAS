#!/usr/bin/env python3
"""
XNNKAS - X-ray Microdiffraction Machine Learning Analysis Software
---------------------------------------------------------------------
This script handles the generation, processing, and training for overlapped 
Laue patterns extracted from X-ray microdiffraction data. It performs the 
following key steps:

1. Data Preparation:
   - Generates lists of Laue patterns. Each element in these lists is an 
     instance of the LauePattern class (defined in Laue_pattern.py). Each 
     LauePattern object contains measurement points, HKLs, and associated metadata.
   - Combines individual LauePattern lists (one per key material) into a single 
     list (all_lauelists). This combined list is used to generate overlapped 
     Laue patterns via the function 'generate_overlap_hist_hkls'.
   - The function 'generate_overlap_hist_hkls' randomly selects patterns and overlaps 
     them together. The parameter 'max_overlapped' (set internally as 5) defines the 
     number of patterns to overlap, while 'max_hkl' specifies the maximum HKL value 
     to consider for prediction in the model.

2. Data Processing:
   - Each LauePattern instance is pre-processed by:
       • Synchronizing HKLs with measurement points (reducing duplicates by 
         simplifying HKLs).
       • Applying symmetry operations based on the specified symmetry group.
       • Filtering HKLs to include only those with values below a defined limit 
         (max_hkl).

3. Model Training:
   - A neural network with skip connections is built using the processed overlap 
     histograms as input.
   - The model is trained for a fixed number of epochs, with class weights computed 
     based on the frequency of each encoded HKL to address class imbalance.

4. Saving Results:
   - After training, the script extracts model weights from a specific layer and 
     saves these weights along with the unique HKL mapping for future inference or analysis.

Note:
   - The LauePattern class (see Laue_pattern.py) also contains methods for noise 
     addition, point augmentation, and peak removal, which help simulate variations 
     in experimental data.
     
Author: KHChan
Created on: Wed Apr 24 15:28:49 2024
"""

import os
import numpy as np
from collections import Counter
import logging

# Local imports from the project
from utils.data_generation import generate_overlap_hist_hkls
from utils.generate_hkl import generate_laue_hkl
from utils.symmetry import Symmetry
from utils.neural_networks import build_network_with_skip_connections

# External imports from LaueTools
from LaueTools.CrystalParameters import Prepare_Grain
from LaueTools.dict_LaueTools import dict_Materials

# Configure logging to display info messages.
logging.basicConfig(level=logging.INFO)

def prepare_lauelists(key_materials, syms, max_hkl, nb_patterns, valid_flag=False):
    """
    Generate and prepare Laue pattern lists for training or validation.
    
    Each Laue pattern is an instance of the LauePattern class. The individual 
    lists (one per key material) are combined into a single list (all_lauelists) 
    which is used to generate overlapped histograms via generate_overlap_hist_hkls.
    
    Args:
        key_materials (list): List of key material names.
        syms (list): List of symmetry operations corresponding to each material.
        max_hkl (int): Maximum hkl value for filtering.
        nb_patterns (int): Number of patterns to generate.
        valid_flag (bool): Flag indicating if data is for validation. Default is False.
        
    Returns:
        tuple: (histograms, hkls, wavelengths) generated from the combined Laue patterns.
    """
    # Generate Laue pattern lists for each material using the provided key_material and symmetry.
    lauelists = [
        generate_laue_hkl(key_material=mat, valid=valid_flag, symmetry=sym) 
        for mat, sym in zip(key_materials, syms)
    ]
    
    # Combine all LauePattern instances into one list for generating overlap histograms.
    all_lauelists = [laue for sublist in lauelists for laue in sublist]
    
    # Pre-process each LauePattern:
    #  - Synchronize HKLs with measurement points (reducing duplicates)
    #  - Apply symmetry operations based on the symmetry group
    #  - Filter HKLs to include only those below the specified max_hkl value.
    for laue in all_lauelists:
        laue.synchronize_hkls_with_points()
        laue.apply_symmetry_operations()
        laue.filter_points_based_on_hkl(max_hkl)
    
    # Generate overlapped histograms, HKLs, and wavelengths.
    # Note: generate_overlap_hist_hkls randomly selects Laue patterns and overlaps them,
    # where 'max_overlapped' is fixed at 5 and 'max_hkl' limits the HKL values for prediction.
    hists, hkls, wavelengths = generate_overlap_hist_hkls(
        all_lauelists, nb_patterns=nb_patterns, max_overlapped=5, max_hkl=max_hkl
    )
    return hists, hkls, wavelengths

def encode_hkls(hkls):
    """
    Encode HKLs into integer labels and calculate class weights.
    
    Args:
        hkls (iterable): List or array of HKL tuples.
    
    Returns:
        tuple: (encoded_hkls, unique_hkls, class_weights, hkl_to_index)
            - encoded_hkls: Array of integer labels for each HKL.
            - unique_hkls: Sorted list of unique HKL tuples.
            - class_weights: Dictionary mapping each class to its weight.
            - hkl_to_index: Mapping from HKL tuple to integer label.
    """
    # Obtain unique HKL tuples and sort them.
    unique_hkls = sorted(set(hkls))
    # Create mapping from HKL tuple to integer index.
    hkl_to_index = {hkl: idx for idx, hkl in enumerate(unique_hkls)}
    # Encode the HKLs using the mapping.
    encoded_hkls = np.array([hkl_to_index[hkl] for hkl in hkls], dtype=int)
    # Calculate class weights to address class imbalance.
    class_counts = Counter(encoded_hkls)
    total_counts = sum(class_counts.values())
    class_weights = {cls: total_counts / count for cls, count in class_counts.items()}
    return encoded_hkls, unique_hkls, class_weights, hkl_to_index

def build_and_train_model(train_hists, train_hkls, valid_hists, valid_hkls, unique_hkls, class_weights):
    """
    Build the neural network model with skip connections and train it.
    
    Args:
        train_hists (ndarray): Training histograms.
        train_hkls (ndarray): Encoded training HKL labels.
        valid_hists (ndarray): Validation histograms.
        valid_hkls (ndarray): Encoded validation HKL labels.
        unique_hkls (list): List of unique HKL tuples.
        class_weights (dict): Weights for each class.
    
    Returns:
        model: Trained neural network model.
    """
    # Build the network using the training histogram shape as input dimension.
    model = build_network_with_skip_connections(
        input_dim=train_hists.shape[1],
        num_layers=1,
        num_classes=len(unique_hkls)
    )
    # Train the model with training data and validate on validation data.
    model.fit(train_hists, train_hkls, epochs=5, class_weight=class_weights,
              validation_data=(valid_hists, valid_hkls))
    return model

def save_model_results(model, key_material, unique_hkls, output_dir='model'):
    """
    Save the trained model's weights and HKL mapping to disk.
    
    Args:
        model: Trained neural network model.
        key_material (str): Material name used as file prefix.
        unique_hkls (list): List of unique HKL tuples.
        output_dir (str): Directory to save the model weights and HKLs.
    """
    # Ensure the output directory exists.
    os.makedirs(output_dir, exist_ok=True)
    # Extract weights from a specific layer (here, layer index 1).
    weight1, weight2 = model.get_layer(index=1).get_weights()
    # Save the weight matrices and unique HKL mapping as numpy arrays.
    np.save(os.path.join(output_dir, f'{key_material}_weight1.npy'), weight1)
    np.save(os.path.join(output_dir, f'{key_material}_weight2.npy'), weight2)
    np.save(os.path.join(output_dir, f'{key_material}_hkls.npy'), np.array(unique_hkls))
    logging.info(f"Weights and HKLs saved for material: {key_material}")

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Configuration Section
    # -------------------------------------------------------------------------
    key_materials = ["BaTiO3"]           # Material(s) to analyze
    syms = [Symmetry.cubic]         # Corresponding symmetry groups
    max_hkl = 8                          # Maximum allowed hkl value for filtering
    nb_patterns_train = 200              # Number of training patterns to generate
    nb_patterns_valid = 20               # Number of validation patterns to generate
    valid_flag = True                   # Indicates whether the generated patterns are for validation

    # -------------------------------------------------------------------------
    # Data Preparation Section
    # -------------------------------------------------------------------------
    # Generate training data: LauePattern objects are created and combined.
    # The combined list is then used to generate overlap histograms, HKLs, and wavelengths.
    train_hists, train_hkls_raw, _ = prepare_lauelists(key_materials, syms, max_hkl, nb_patterns_train)
    # Encode the HKLs to integer labels and compute class weights.
    train_hkls, unique_hkls, class_weights, hkl_to_index = encode_hkls(train_hkls_raw)
    
    # Generate validation data using the same process.
    valid_hists, valid_hkls_raw, _ = prepare_lauelists(key_materials, syms, max_hkl, nb_patterns_valid, valid_flag=valid_flag)
    # Encode validation HKLs using the same mapping from training data.
    valid_hkls = np.array([hkl_to_index[hkl] for hkl in valid_hkls_raw], dtype=int)

    # -------------------------------------------------------------------------
    # Model Training Section
    # -------------------------------------------------------------------------
    # Build and train the neural network model with the prepared data.
    model = build_and_train_model(train_hists, train_hkls, valid_hists, valid_hkls, unique_hkls, class_weights)

    # -------------------------------------------------------------------------
    # Saving Results Section
    # -------------------------------------------------------------------------
    # Save the trained model's weights and HKL mapping for future use.
    save_model_results(model, key_materials[0], unique_hkls)
