# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 11:54:27 2023

@author: KHChan
"""
from tifffile import TiffFile
from os import listdir
from os.path import join
import concurrent.futures
import numpy as np

def read_tiff_energy(file_path):
    with TiffFile(file_path) as tif:
        description = tif.pages[0].description.strip().split('\n')
        try:
            energy = float(description[13].split()[4])
        except:
            energy = 0
        img = tif.pages[0].asarray()
    return img, energy

def process_folder(input_folder):
    file_list = [f for f in listdir(input_folder) if f.endswith('.tif')]
    input_file_paths = [join(input_folder, filename) for filename in file_list]

    energies = []
    images = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(read_tiff_energy, path): path for path in input_file_paths}
        
        results = []
        for future in concurrent.futures.as_completed(futures):
            img, energy = future.result()
            results.append((energy, np.rot90(img, k=1, axes=(1,0))))

    # Sort results based on energy values
    results.sort(key=lambda x: x[0])

    # Unpack sorted energies and images
    energies, images = zip(*results)  # This will create tuples, convert them to lists if needed
    energies = list(energies)
    images = list(images)

    min_energy = min(energies)
    max_energy = max(energies)

    return min_energy, max_energy, energies, images
"""
# Example usage
start = time()
input_folder = 'C:/Users/khchan/Downloads/S184_RT_002/'
min_energy, max_energy, images = process_folder(input_folder)
end = time()

print(f"Min energy: {min_energy}, Max energy: {max_energy}")
print(f"Time taken: {end - start} seconds")

"""