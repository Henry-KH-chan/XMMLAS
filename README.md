# XMMKAS: X-ray Microdiffraction Machine Learning Analysis Software

## Overview

**XMMKAS** is a lightweight machine learning tool designed to analyze Laue patterns from X-ray microdiffraction experiments. Built on top of [LaueTools](https://pypi.org/project/LaueTools/) and [lauetoolsnn](https://pypi.org/project/lauetoolsnn/), XMMKAS uses a simple neural network to efficiently solve Laue patterns. The project is optimized for faster training by employing a single-layer neural network implemented via matrix multiplication, significantly speeding up both training and convergence. Additionally, the indexing procedure is enhanced by leveraging the intensity of the peaks—ranking and iteratively feeding them to the network until successful indexing is achieved.

## Intended Audience

This project is aimed at researchers, scientists, and practitioners who need to analyze Laue images, particularly those acquired at the ALS Advanced Light Source micro focus beamline 12.3.2. While the tool is optimized for this specific setup, it can be adapted for other configurations with further fine-tuning.

## Installation & Environment Setup

### For Processing Only

If you only need to process Laue images (using `process_laue.py`), the core package installation via `setup.py` is sufficient.

1. **Install in Editable Mode:**

   ```bash
   pip install -e .
2. **Alternatively, Install the Minimal Dependencies:**

    - Create a file named `requirements_process.txt` with the following content:

        ```txt
        numpy
        tifffile
        scipy
        scikit-image
        matplotlib
        h5py
        ```

    - Then install them by running:

        ```bash
        pip install -r requirements_process.txt
        ```

### For Training

For training functionality (using `training.py`), TensorFlow is required. It is recommended to use TensorFlow 2.10, which is compatible with CUDA 11.2 and cuDNN 8.1 for GPU training. The CPU version is also effective given the optimized single-layer network.

- **For GPU Training:**

    ```bash
    pip install tensorflow_gpu==2.10.0
    ```

- **For CPU Training:**

    ```bash
    pip install tensorflow==2.10.0
    ```

### Using a Conda Environment

Alternatively, you can create a conda environment using the provided `environment.yaml` file. Save the following content as `environment.yaml`:

```yaml
name: xmmkas_env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - numpy
  - tifffile
  - scipy
  - scikit-image
  - matplotlib
  - h5py
  # Uncomment one of the following lines for training functionality:
  # For GPU support (requires appropriate CUDA/cuDNN installation):
  # - tensorflow-gpu=2.10.0
  # For CPU-only training:
  # - tensorflow=2.10.0
  ```

To create and activate the environment, run:
```bash
conda env create -f environment.yaml 
conda activate xmmkas_env
```

### Usage
**Processing Laue Images**
The process_laue.py script processes raw Laue images, which are typically acquired using a Pilatus detector (1043 x 981 pixels). The geometry configuration is generally determined using tools like XMAS or LaueTools with a standard sample.
Put your images inside the folder data and change the directory inside the script.

To process images, run:
```bash
python src/xmmkas/process_laue.py
```
Processed results—such as orientation matrices and additional details will be saved in the data/processed/ directory.

**Training the Model**
The training.py script trains the neural network using generated overlap histograms and encoded HKL data. Ensure your environment includes TensorFlow and that your system meets the CUDA/cuDNN requirements (if using GPU). CPU can also run properly. 

To train the model, change the parameter inside the scripts and run:
```bash
python src/xmmkas/training.py
```

### License
This project is licensed under the [MIT License](https://mit-license.org/).

### Contributing
Contributions are welcome! Please open an issue or submit a pull request for any bug fixes, improvements, or suggestions. For detailed guidelines, refer to the CONTRIBUTING.md file.

### Contact
For questions or support, please contact me at [email](mailto:tkp203059@gmail.com) or open an issue on GitHub.

### Acknowledgements
[LaueTools](https://pypi.org/project/LaueTools/)

[lauetoolsnn](https://pypi.org/project/lauetoolsnn/)

The ALS Advanced Light Source micro focus beamline 12.3.2 team (Dr. Nobumichi Tamura) for their experimental support.


