"""Constants and configurations for the application."""
# Author: Samarth Sanjay Jadhav
# GitHub: https://github.com/Samarth-Sanjay-Jadhav
# Course: Computer Vision
# Project: Smart Recycling Bin Classifier
import yaml
from pathlib import Path

# Load configuration
CONFIG_PATH = Path("config.yaml")
with open(CONFIG_PATH, 'r') as f:
    CONFIG = yaml.safe_load(f)

# Categories
CATEGORIES = CONFIG['categories']
NUM_CLASSES = len(CATEGORIES)

# Bin mapping
BIN_MAPPING = CONFIG['bins']

# Image processing parameters
IMAGE_SIZE = tuple(CONFIG['image_processing']['target_size'])
CANNY_THRESHOLD1 = CONFIG['image_processing']['canny_threshold1']
CANNY_THRESHOLD2 = CONFIG['image_processing']['canny_threshold2']
MORPHOLOGY_KERNEL_SIZE = CONFIG['image_processing']['morphology_kernel_size']
KMEANS_CLUSTERS = CONFIG['image_processing']['kmeans_clusters']

# Model parameters
MODEL_PATH = CONFIG['model']['model_path']
FEATURE_SCALER_PATH = CONFIG['model']['feature_scaler_path']

# App settings
MAX_UPLOAD_SIZE = CONFIG['app']['max_upload_size_mb'] * 1024 * 1024
SUPPORTED_FORMATS = CONFIG['app']['supported_formats']
CONFIDENCE_THRESHOLD = CONFIG['app']['confidence_threshold']

# Color mapping for display
COLOR_MAP = {
    'Plastic': '#3498db',     # Blue
    'Metal': '#f1c40f',       # Yellow
    'Paper': '#2ecc71',       # Green
    'Cardboard': '#e67e22',   # Orange
    'Glass': '#8b4513',       # Brown
    'Unknown': '#95a5a6'      # Gray
}

# Emoji mapping
EMOJI_MAP = {
    'Plastic': '🔵',
    'Metal': '🟡',
    'Paper': '🟢',
    'Cardboard': '📦',
    'Glass': '🟤',
    'Unknown': '❓'
}