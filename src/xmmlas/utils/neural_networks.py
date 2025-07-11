# -*- coding: utf-8 -*- 
"""
Created on Wed Jul 24 11:26:01 2024

@author: KHChan
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate, Flatten
from tensorflow.keras.optimizers import Adam

def build_network(input_dim, num_classes):
    """
    Build a simple feedforward neural network with two hidden layers for multi-class classification.
    
    Args:
        input_dim (int): Number of features in the input.
        num_classes (int): Number of output classes.
    
    Returns:
        model (tensorflow.keras.Model): Compiled Keras model.
    """
    # Define the input layer
    inputs = Input(shape=(input_dim,))
    
    # First hidden layer with 500 neurons, ReLU activation, batch normalization, and dropout.
    x = Dense(500, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Second hidden layer with the same configuration.
    x = Dense(500, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Output layer with softmax activation for multi-class classification.
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create and compile the model.
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def build_network_with_skip_connections_and_adapted_angle(input_dim, angle_dim, num_classes):
    """
    Build a neural network with two input branches: one for the main features and one for angle features.
    The angle branch is processed through Dense layers and then concatenated with the main input.
    The network employs skip connections to improve feature propagation.
    
    Args:
        input_dim (int): Dimension of the main input.
        angle_dim (int): Dimension of the angle input.
        num_classes (int): Number of output classes.
    
    Returns:
        model (tensorflow.keras.Model): Compiled Keras model with two inputs.
    """
    # Define the main input layer.
    main_input = Input(shape=(input_dim,), name='main_input')
    
    # Define the angle input layer and process it through a series of Dense layers.
    angle_input = Input(shape=(angle_dim,), name='angle_input')
    angle_branch = Dense(128, activation='relu')(angle_input)
    angle_branch = BatchNormalization()(angle_branch)
    angle_branch = Dense(64, activation='relu')(angle_branch)
    angle_branch = BatchNormalization()(angle_branch)
    angle_branch = Dense(32, activation='relu')(angle_branch)
    angle_branch = BatchNormalization()(angle_branch)
    
    # Concatenate the processed angle branch with the main inputs.
    combined_inputs = Concatenate(name='concat_inputs')([main_input, angle_branch])
    
    # First block with skip connection
    x = Dense(500, activation='relu')(combined_inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(500, activation='relu')(x)
    x = BatchNormalization()(x)
    
    # Create skip connection from the combined inputs.
    skip_x = Concatenate()([combined_inputs, x])
    
    # Second block with a skip connection
    x = Dropout(0.5)(skip_x)
    x = Dense(500, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(500, activation='relu')(x)
    x = BatchNormalization()(x)
    
    # Combine skip connections from previous layers.
    skip_x = Concatenate()([skip_x, x])
    
    # Output layer with softmax activation for multi-class classification.
    outputs = Dense(num_classes, activation='softmax')(skip_x)
    
    # Create and compile the model.
    model = Model(inputs=[main_input, angle_input], outputs=outputs)
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def build_network_with_ten_class_outputs(input_dim, num_classes_list):
    """
    Build a neural network with 10 output branches.
    
    Args:
        input_dim (int): Number of input features.
        num_classes_list (list of int): A list of number of classes for each of the 10 outputs.
    
    Returns:
        model (tensorflow.keras.Model): Compiled Keras model with 10 outputs.
    """
    # Define the input layer.
    inputs = Input(shape=(input_dim,))
    
    # Create 10 independent output layers.
    outputs = []
    for i in range(len(num_classes_list)):
        output = Dense(num_classes_list[i], activation='softmax', name=f'classification_output_{i+1}')(inputs)
        outputs.append(output)
    
    # Create the model with 10 outputs.
    model = Model(inputs=inputs, outputs=outputs)
    
    # Create loss and metric dictionaries for each output.
    loss_dict = {f'classification_output_{i+1}': 'sparse_categorical_crossentropy' for i in range(len(num_classes_list))}
    metrics_dict = {f'classification_output_{i+1}': 'accuracy' for i in range(len(num_classes_list))}
    
    # Compile the model with these dictionaries.
    model.compile(optimizer=Adam(), loss=loss_dict, metrics=metrics_dict)
    
    return model

def build_network_with_skip_connections(input_dim, num_layers, num_classes):
    """
    Build a simplified neural network with skip connections.
    
    This version only uses the main input without an extra angle branch. The network 
    architecture is minimal: input followed directly by an output layer.
    
    Args:
        input_dim (int): Dimension of the input features.
        num_layers (int): This parameter is not used in the current implementation but could be
                          used to dynamically set the depth of the network.
        num_classes (int): Number of output classes.
    
    Returns:
        model (tensorflow.keras.Model): Compiled Keras model.
    """
    # Define the input layer.
    inputs = Input(shape=(input_dim,))
    
    # The current implementation does not include hidden layers; it goes directly to output.
    outputs = Dense(num_classes, activation='softmax')(inputs)
    
    # Create and compile the model.
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Example usage (commented out):
"""
# Example usage:
input_dim = 900  # Dimension of the main input
angle_dim = 1    # Dimension for the angle input (single value)
num_classes = 10 # Number of output classes for the adapted network

model = build_network_with_skip_connections_and_adapted_angle(input_dim, angle_dim, num_classes)
model.summary()
"""
