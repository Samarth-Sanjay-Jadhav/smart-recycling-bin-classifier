"""Image segmentation functions."""

import cv2
import numpy as np
from typing import Tuple

def kmeans_segmentation(image: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment image using K-means clustering.
    
    Args:
        image: Input image (BGR)
        k: Number of clusters
        
    Returns:
        Tuple of (segmented image, labels)
    """
    # Reshape image to 2D array of pixels
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert centers to uint8
    centers = np.uint8(centers)
    
    # Create segmented image
    segmented = centers[labels.flatten()]
    segmented = segmented.reshape(image.shape)
    
    return segmented, labels.reshape(image.shape[:2])

def watershed_segmentation(image: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Segment image using Watershed algorithm.
    
    Args:
        image: Input image (BGR)
        edges: Edge map
        
    Returns:
        Segmented image with markers
    """
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create marker image
    markers = np.zeros(edges.shape, dtype=np.int32)
    
    # Draw contours on markers
    for i, contour in enumerate(contours):
        cv2.drawContours(markers, [contour], 0, i + 1, -1)
    
    # Apply watershed
    marked_image = image.copy()
    cv2.watershed(marked_image, markers)
    
    # Create output image
    output = image.copy()
    output[markers == -1] = [0, 0, 255]  # Mark boundaries in red
    
    return output

def contour_analysis(image: np.ndarray, edges: np.ndarray) -> dict:
    """
    Analyze contours in the image.
    
    Args:
        image: Input image
        edges: Edge map
        
    Returns:
        Dictionary with contour information
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    info = {
        'num_objects': len(contours),
        'contours': contours,
        'areas': [],
        'circularity': []
    }
    
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            info['circularity'].append(circularity)
        
        info['areas'].append(area)
    
    return info

def draw_contours_on_image(image: np.ndarray, edges: np.ndarray, 
                           color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Draw contours on image.
    
    Args:
        image: Input image
        edges: Edge map
        color: Color for contours (BGR)
        
    Returns:
        Image with drawn contours
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    output = image.copy()
    cv2.drawContours(output, contours, -1, color, 2)
    
    return output