"""Feature extraction for classification."""

import cv2
import numpy as np
from typing import List

def extract_color_features(image: np.ndarray) -> List[float]:
    """
    Extract color-based features from image.
    
    Args:
        image: Input image (BGR)
        
    Returns:
        List of color features
    """
    # Convert to HSV for color features
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    features = []
    
    # Mean and std of each channel
    for i in range(3):
        features.append(np.mean(hsv[:, :, i]))
        features.append(np.std(hsv[:, :, i]))
    
    # Color histogram
    hist_h = cv2.calcHist([hsv], [0], None, [256], [0, 256])
    features.extend(hist_h.flatten()[:8])  # Use first 8 bins
    
    return features

def extract_texture_features(image: np.ndarray) -> List[float]:
    """
    Extract texture-based features using Laplacian.
    
    Args:
        image: Input image
        
    Returns:
        List of texture features
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Laplacian operator (edge intensity)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    features = [
        np.mean(laplacian),
        np.std(laplacian),
        np.max(laplacian)
    ]
    
    return features

def extract_shape_features(image: np.ndarray, edges: np.ndarray) -> List[float]:
    """
    Extract shape-based features from contours.
    
    Args:
        image: Input image
        edges: Edge map
        
    Returns:
        List of shape features
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return [0] * 5
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    
    features = [area, perimeter]
    
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter ** 2)
        features.append(circularity)
    else:
        features.append(0)
    
    # Fit ellipse if possible
    if len(largest_contour) >= 5:
        ellipse = cv2.fitEllipse(largest_contour)
        features.extend([ellipse[1][0], ellipse[1][1]])  # Major and minor axes
    else:
        features.extend([0, 0])
    
    return features

def extract_all_features(image: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Extract all features from image.
    
    Args:
        image: Input image
        edges: Edge map
        
    Returns:
        Feature vector (1D array of 22 features)
    """
    color_features = extract_color_features(image)
    texture_features = extract_texture_features(image)
    shape_features = extract_shape_features(image, edges)
    
    all_features = color_features + texture_features + shape_features
    
    return np.array(all_features)  # Return as 1D array