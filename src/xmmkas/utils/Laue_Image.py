# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 10:11:01 2024

@author: KHChan
"""

import sys
sys.path.append("..")
import numpy as np
import tifffile as tiff
from scipy import ndimage
from skimage.feature import peak_local_max

# Attempt relative imports first; fall back to absolute if that fails
try:
    from .distance import compute_angular_distance  # Updated name to match previous references
    from .pixel22theta import convert_pixel_to_theta_chi_XMAS  # Updated name to reflect clarity
except ImportError:
    from distance import compute_angular_distance  # Updated name to match previous references
    from pixel22theta import convert_pixel_to_theta_chi_XMAS

class LaueImage:
    """
    A class to handle Laue image loading, processing, and feature extraction.
    This includes background subtraction, local maxima detection, pixel-to-angle
    conversions, and histogram calculations of angular distances.
    """

    def __init__(self, file_path, mask = None,
                 camera_dimensions=(1043, 981),
                 detector_pixel_sizes=(179, 168.387),
                 sample_detector_distance=150.89621,
                 beam_center=(527.90134, 200.26812),
                 detector_tilt=(0, -0.21886, 0.59401),
                 background_filter_size=11):
        """
        Initialize the LaueImage class with image properties and processing parameters.

        Args:
            file_path (str): Path to the TIFF image file.
            camera_dimensions (tuple): Dimensions of the camera in pixels (width, height).
            detector_pixel_sizes (tuple): Size of one pixel on the detector in mm (width, height).
            sample_detector_distance (float): Distance from the sample to the detector in mm.
            beam_center (tuple): Pixel coordinates of the beam center (x, y).
            detector_tilt (tuple): Tilt angles of the detector (x, y, z).
            background_filter_size (int): Size of the filter for background subtraction.
        """
        self.file_path = file_path
        self.camera_dimensions = camera_dimensions
        self.detector_pixel_sizes = detector_pixel_sizes
        self.sample_detector_distance = sample_detector_distance
        self.beam_center = beam_center
        self.detector_tilt = detector_tilt
        self.background_filter_size = 11
        if background_filter_size > 3:
            self.background_filter_size = 2*np.round(background_filter_size//2)+1
        if mask is not None:
            self.mask = mask

        self.image = None
        self.local_maxima_points = None
        self.twotheta_chi_points = None
        self.histograms = []
        self.intensity = None
        
        # Load and preprocess the image upon initialization
        self.load_image()

    def load_image(self):
        """Load, background-subtract, and rotate the Laue image."""
        try:
            with tiff.TiffFile(self.file_path) as tif:
                self.image = tif.asarray()
                
                # Apply background subtraction to multiple ROIs
                self.subtract_background(size = self.background_filter_size)
                # Clipping negative values
                self.image[self.image < 0] = np.abs(self.image[self.image < 0])

                # Rotate and cast the image appropriately
                if np.max(self.image) > 65535:
                    # Scale down and rotate if values exceed 16-bit range
                    self.image = np.rot90(np.uint16(self.image * 0.031249523162841797), k=1, axes=(1, 0))
                else:
                    self.image = np.rot90(np.uint16(self.image), k=1, axes=(1, 0))

        except FileNotFoundError:
            print(f"File {self.file_path} not found.")
        except Exception as e:
            print(f"An error occurred while loading the image: {e}")

    def subtract_background(self, size = 11):
        """
        Subtract background from a given image segment using a uniform filter.
        
        Args:
            image (np.ndarray): The image segment on which to perform background subtraction.
            size (int): The size of the uniform filter.

        Returns:
            np.ndarray: The background-subtracted image segment.
        """
        if self.image is not None:
            
            if self.mask is None:
                self.mask = np.zeros(self.image.shape, dtype=bool)
            bg = self.image * 1.
            ratio = (np.ones(bg.shape)-ndimage.uniform_filter(np.double(self.mask),size))
            
            for i in range(5):
                newbg = bg * 1.
                newbg[self.mask] = 0
                newbg[~self.mask] = (ndimage.uniform_filter(np.double(bg),size)[~self.mask]/ratio[~self.mask])
                bg[bg>(newbg)] = newbg[bg>(newbg)]
                bg[self.mask] = 0
            newbg = bg * 1.
            newbg[self.mask] = 0
            newbg[~self.mask] = (ndimage.uniform_filter(np.double(bg),size)[~self.mask]/ratio[~self.mask])
            newbg[self.mask] = 0
            self.image = self.image - bg
        else:
            print("No image is loaded for background subtraction.")

    def find_local_maxima(self, min_distance=5, threshold_mul=10):
        """
        Identify local maxima in the processed image. Use a Gaussian filter for smoothing,
        find local maxima, and calculate intensities.

        Args:
            min_distance (int): Minimum number of pixels between peaks.
            threshold (int): Absolute intensity threshold for peaks.
        """
        if self.image is not None:
            # Smooth the image before peak detection
            image_smoothed = ndimage.gaussian_filter(self.image, sigma=1)
            local_maxima = peak_local_max(image_smoothed, min_distance=min_distance, threshold_abs = threshold_mul * np.mean(self.image))
            self.local_maxima_points = local_maxima
            
            # Calculate the intensity of each peak
            window_size = 4
            intensity_sums = []
            for y, x in local_maxima:
                y_min = max(0, y - window_size // 2)
                y_max = min(self.image.shape[0], y + window_size // 2 + 1)
                x_min = max(0, x - window_size // 2)
                x_max = min(self.image.shape[1], x + window_size // 2 + 1)

                window = self.image[y_min:y_max, x_min:x_max]
                intensity_sum = np.sum(window)
                intensity_sums.append(intensity_sum)

            self.intensity = np.array(intensity_sums)
        else:
            print("No processed image available. Please load the image first.")

    def convert_pixels_to_theta_chi(self):
        """
        Convert the local maxima pixel coordinates into theta-chi angles.
        Requires that local maxima have been identified first.
        """
        if self.local_maxima_points is not None and len(self.local_maxima_points) > 0:
            x_coords = self.local_maxima_points[:, 1]
            y_coords = self.local_maxima_points[:, 0]
            twotheta_chi_points = convert_pixel_to_theta_chi_XMAS(
                x_coords, y_coords, self.camera_dimensions,
                self.detector_pixel_sizes, self.sample_detector_distance,
                self.beam_center, self.detector_tilt
            )
            self.twotheta_chi_points = np.array(twotheta_chi_points).T
        else:
            print("No local maxima found. Please run 'find_local_maxima' first.")

    def calculate_histograms_of_distances(self):
        """
        Compute the angular distance between all detected points and produce histograms of these distances.
        This requires that the theta-chi coordinates have been computed.
        """
        if self.twotheta_chi_points is not None and len(self.twotheta_chi_points) > 0:
            distances = compute_angular_distance(self.twotheta_chi_points, self.twotheta_chi_points)

            # Set distances close to zero to a large number to avoid self-comparison issues
            distances[np.abs(distances) < 0.1] = 300

            # Compute histograms for each point
            self.histograms = np.array([np.histogram(dist, bins=900, range=(0, 90))[0] for dist in distances])
        else:
            print("No theta-chi points available. Please run 'convert_pixels_to_theta_chi' first.")