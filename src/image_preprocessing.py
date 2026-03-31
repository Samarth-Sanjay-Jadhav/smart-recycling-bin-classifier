"""Image preprocessing functions."""

import cv2
import numpy as np
from typing import Tuple
from src.constants import IMAGE_SIZE, CANNY_THRESHOLD1, CANNY_THRESHOLD2

def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Enhance image contrast using histogram equalization.
    
    Args:
        image: Input image (BGR)
        
    Returns:
        Enhanced image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Equalize L channel
    l_channel = cv2.equalizeHist(l_channel)
    
    # Merge back
    enhanced_lab = cv2.merge([l_channel, a_channel, b_channel])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced

def adaptive_histogram_equalization(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        image: Input image (BGR)
        clip_limit: Contrast limit
        
    Returns:
        Enhanced image
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    
    enhanced_lab = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

def edge_detection(image: np.ndarray) -> np.ndarray:
    """
    Detect edges using Canny edge detection.
    
    Args:
        image: Input image (BGR)
        
    Returns:
        Edge map
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, CANNY_THRESHOLD1, CANNY_THRESHOLD2)
    
    return edges

def morphological_operations(image: np.ndarray) -> np.ndarray:
    """
    Apply morphological operations to clean up image.
    
    Args:
        image: Input image (binary)
        
    Returns:
        Cleaned image
    """
    from src.constants import MORPHOLOGY_KERNEL_SIZE
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                       (MORPHOLOGY_KERNEL_SIZE, MORPHOLOGY_KERNEL_SIZE))
    
    # Closing (dilation followed by erosion) - fills small holes
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Opening (erosion followed by dilation) - removes small objects
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return opened

def preprocess_image(image: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Complete preprocessing pipeline.
    
    Args:
        image: Input image
        
    Returns:
        Tuple of (preprocessed image, processing steps dict)
    """
    steps = {}
    
    # Resize
    resized = cv2.resize(image, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    steps['original'] = image
    steps['resized'] = resized
    
    # Enhance contrast
    enhanced = adaptive_histogram_equalization(resized)
    steps['enhanced'] = enhanced
    
    # Edge detection
    edges = edge_detection(enhanced)
    steps['edges'] = edges
    
    # Morphological operations
    cleaned = morphological_operations(edges)
    steps['morphological'] = cleaned
    
    return resized, steps