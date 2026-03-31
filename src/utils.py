"""Utility functions for the application."""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import streamlit as st
from typing import Tuple, List, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_image(image_file) -> Optional[np.ndarray]:
    """
    Load image from uploaded file.
    
    Args:
        image_file: Uploaded image file from Streamlit
        
    Returns:
        Image as numpy array or None if error
    """
    try:
        img = Image.open(image_file)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.error(f"Error loading image: {str(e)}")
        st.error(f"Error loading image: {str(e)}")
        return None

def validate_image(image: np.ndarray, max_size_mb: float = 10) -> bool:
    """
    Validate image format and size.
    
    Args:
        image: Image as numpy array
        max_size_mb: Maximum allowed size in MB
        
    Returns:
        True if valid, False otherwise
    """
    if image is None:
        return False
    
    # Check dimensions
    if image.shape[0] < 50 or image.shape[1] < 50:
        st.warning("Image too small. Minimum 50x50 pixels required.")
        return False
    
    if image.shape[0] > 4000 or image.shape[1] > 4000:
        st.warning("Image too large. Maximum 4000x4000 pixels.")
        return False
    
    return True

def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image to target size while maintaining aspect ratio.
    
    Args:
        image: Input image
        target_size: Target (height, width)
        
    Returns:
        Resized image
    """
    return cv2.resize(image, (target_size[1], target_size[0]), 
                      interpolation=cv2.INTER_AREA)

def display_side_by_side(col1, col2, img1: np.ndarray, img2: np.ndarray, 
                         title1: str, title2: str):
    """
    Display two images side by side in Streamlit.
    
    Args:
        col1: Streamlit column 1
        col2: Streamlit column 2
        img1: First image
        img2: Second image
        title1: Title for first image
        title2: Title for second image
    """
    with col1:
        st.image(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), caption=title1, use_column_width=True)
    with col2:
        st.image(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), caption=title2, use_column_width=True)

def get_bin_recommendation(predicted_class: str) -> dict:
    """
    Get bin recommendation for predicted waste class.
    
    Args:
        predicted_class: Predicted waste category
        
    Returns:
        Dictionary with bin information
    """
    from src.constants import BIN_MAPPING, EMOJI_MAP
    
    bin_info = BIN_MAPPING.get(predicted_class, {})
    return {
        'bin_number': bin_info.get('bin_number', 'Unknown'),
        'color': bin_info.get('color', 'Unknown'),
        'emoji': EMOJI_MAP.get(predicted_class, '❓'),
        'category': predicted_class
    }

def format_confidence(confidence: float) -> str:
    """Format confidence score as percentage."""
    return f"{confidence * 100:.2f}%"

def log_detection(category: str, confidence: float, image_name: str):
    """Log detection results."""
    logger.info(f"Detection: {category} ({confidence:.2%}) - {image_name}")