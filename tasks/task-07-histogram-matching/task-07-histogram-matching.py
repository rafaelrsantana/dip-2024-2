# histogram_matching_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `match_histograms_rgb(source_img, reference_img)` that receives two RGB images
(as NumPy arrays with shape (H, W, 3)) and returns a new image where the histogram of each RGB channel 
from the source image is matched to the corresponding histogram of the reference image.

Your task:
- Read two RGB images: source and reference (they will be provided externally).
- Match the histograms of the source image to the reference image using all RGB channels.
- Return the matched image as a NumPy array (uint8)

Function signature:
    def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray

Return:
    - matched_img: NumPy array of the result image

Notes:
- Do NOT save or display the image in this function.
- Do NOT use OpenCV to apply the histogram match (only for loading images, if needed externally).
- You can assume the input images are already loaded and in RGB format (not BGR).
"""

import cv2 as cv
import numpy as np
import scikitimage as ski

def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:

    source_img = source_img.astype(np.uint8)
    reference_img = reference_img.astype(np.uint8)
    
    matched_img = np.zeros_like(source_img)
    
    for channel in range(3):
        source_channel = source_img[:,:,channel]
        reference_channel = reference_img[:,:,channel]
        
        hist_source, _ = np.histogram(source_channel.flatten(), 256, [0, 256])
        hist_reference, _ = np.histogram(reference_channel.flatten(), 256, [0, 256])
        
        
        cdf_source = hist_source.cumsum()
        cdf_source = cdf_source / cdf_source[-1]  
        
        cdf_reference = hist_reference.cumsum()
        cdf_reference = cdf_reference / cdf_reference[-1]  
        
        
        lookup_table = np.zeros(256, dtype=np.uint8)
        
        for src_level in range(256):
            src_cdf_value = cdf_source[src_level]
            
            idx_min = np.argmin(np.abs(cdf_reference - src_cdf_value))
            lookup_table[src_level] = idx_min
        
        matched_img[:,:,channel] = lookup_table[source_channel]
    
    return matched_img