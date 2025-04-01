import setuptools
import os

# Read the README file for a long description.
this_directory = os.path.abspath(os.path.dirname(__file__))
# with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
#     long_description = f.read()

setuptools.setup(
    name="xmmkas",
    version="1.0.0",
    author="Ka Hung CHAN",
    author_email="tkp203059@gmail.com",
    description="X-ray microdiffraction machine learning analysis software",
    # long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Henry-KH-chan/XMMLAS",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # or your license
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies that are common (if any)
        "numpy",
        "tifffile",
        "scipy",
        "scikit-image",
        "matplotlib",
        "h5py"
    ],
    extras_require={
        "train": [
            "tensorflow_gpu==2.10.0",  # For GPU training; ensure CUDA/cuDNN are installed.
            # Alternatively, you can specify a CPU-only version if desired:
            # "tensorflow==2.10.0",
        ]
    },
    include_package_data=True,
)
