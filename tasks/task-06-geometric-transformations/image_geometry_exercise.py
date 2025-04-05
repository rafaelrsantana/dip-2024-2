# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np

def translate_image(img, shift_x=30, shift_y=30):
    height, width = img.shape
    result = np.zeros_like(img)

    y_valid = min(height - shift_y, height)
    x_valid = min(width - shift_x, width)
    
    if y_valid > 0 and x_valid > 0:
        result[shift_y:, shift_x:] = img[:y_valid, :x_valid]
    
    return result

def rotate_90_clockwise(img):
    return np.flip(img.T, axis=0)

def stretch_horizontally(img, scale=1.5):
    height, width = img.shape
    new_width = int(width * scale)
    result = np.zeros((height, new_width), dtype=img.dtype)
    
    for y in range(height):
        for x in range(new_width):
            src_x = int(x / scale)
            if 0 <= src_x < width:
                result[y, x] = img[y, src_x]
    
    return result

def mirror_horizontally(img):
    return np.fliplr(img)

def barrel_distortion(img, distortion_factor=0.3):
    height, width = img.shape
    result = np.zeros_like(img)
    
    center_x = width / 2
    center_y = height / 2
    
    for y in range(height):
        for x in range(width):
            # Calculate normalized coordinates relative to center
            nx = (x - center_x) / center_x
            ny = (y - center_y) / center_y
            
            # Calculate radius
            r = np.sqrt(nx*nx + ny*ny)
            
            # Apply barrel distortion formula
            distorted_r = r * (1 + distortion_factor * r * r)
            
            # Map back to image coordinates
            if r > 0:
                src_x = int(center_x + nx * (distorted_r / r) * center_x)
                src_y = int(center_y + ny * (distorted_r / r) * center_y)
            else:
                src_x, src_y = x, y
            
            # Copy pixel if source is within bounds
            if 0 <= src_x < width and 0 <= src_y < height:
                result[y, x] = img[src_y, src_x]
    
    return result

def apply_geometric_transformations(img: np.ndarray) -> dict:

    translated = translate_image(img)
    rotated = rotate_90_clockwise(img)
    stretched = stretch_horizontally(img)
    mirrored = mirror_horizontally(img)
    distorted = barrel_distortion(img)
    
    return {
        "translated": translated,
        "rotated": rotated,
        "stretched": stretched,
        "mirrored": mirrored,
        "distorted": distorted
    }