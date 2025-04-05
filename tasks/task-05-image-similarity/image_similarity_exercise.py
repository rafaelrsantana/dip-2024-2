# image_similarity_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `compare_images(i1, i2)` that receives two grayscale images
represented as NumPy arrays (2D arrays of shape (H, W)) and returns a dictionary with the following metrics:

1. Mean Squared Error (MSE)
2. Peak Signal-to-Noise Ratio (PSNR)
3. Structural Similarity Index (SSIM) - simplified version without using external libraries
4. Normalized Pearson Correlation Coefficient (NPCC)

You must implement these functions yourself using only NumPy (no OpenCV, skimage, etc).

Each function should be implemented as a helper function and called inside `compare_images(i1, i2)`.

Function signature:
    def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:

The return value should be like:
{
    "mse": float,
    "psnr": float,
    "ssim": float,
    "npcc": float
}

Assume that i1 and i2 are normalized grayscale images (values between 0 and 1).
"""

import numpy as np

def calculate_mse(i1: np.ndarray, i2: np.ndarray) -> float:
    return np.mean((i1 - i2) ** 2)

def calculate_psnr(mse: float) -> float:
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

def calculate_ssim(i1: np.ndarray, i2: np.ndarray) -> float:

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Compute means
    mu1 = np.mean(i1)
    mu2 = np.mean(i2)
    

    sigma1_sq = np.var(i1)
    sigma2_sq = np.var(i2)
    sigma12 = np.mean((i1 - mu1) * (i2 - mu2))
    
    # SSIM formula
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    
    return numerator / denominator

def calculate_npcc(i1: np.ndarray, i2: np.ndarray) -> float:

    i1_centered = i1 - np.mean(i1)
    i2_centered = i2 - np.mean(i2)
    
    numerator = np.sum(i1_centered * i2_centered)
    denominator = np.sqrt(np.sum(i1_centered**2) * np.sum(i2_centered**2))
    
    if denominator == 0:
        return 0
    
    return numerator / denominator

def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:

    if i1.shape != i2.shape:
        raise ValueError("Images must have the same dimensions")
    
    mse = calculate_mse(i1, i2)
    psnr = calculate_psnr(mse)
    ssim = calculate_ssim(i1, i2)
    npcc = calculate_npcc(i1, i2)
    
    return {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
        "npcc": npcc
    }