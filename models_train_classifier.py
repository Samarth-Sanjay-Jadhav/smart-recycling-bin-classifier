"""Script to train waste classifier model."""
# Author: Samarth Sanjay Jadhav
# GitHub: https://github.com/Samarth-Sanjay-Jadhav
# Course: Computer Vision
# Project: Smart Recycling Bin Classifier
import os
import sys
import numpy as np
import cv2
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))

from src.image_preprocessing import preprocess_image, edge_detection
from src.segmentation import kmeans_segmentation
from src.feature_extraction import extract_all_features
from src.constants import CATEGORIES

def load_dataset(dataset_path: str = "data/dataset"):
    """
    Load waste dataset with folder name mapping.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        Tuple of (images, labels)
    """
    # Map actual folder names to category indices
    # Note: 'trash' folder is excluded - used only for validation/testing
    folder_to_category = {
        'plastic': 0,      # Plastic
        'metal': 1,        # Metal
        'paper': 2,        # Paper
        'cardboard': 3,    # Cardboard
        'glass': 4,        # Glass
    }
    
    images = []
    labels = []
    
    dataset_base = Path(dataset_path)
    
    for folder_name, category_idx in folder_to_category.items():
        folder_path = dataset_base / folder_name
        
        if not folder_path.exists():
            print(f"Warning: Category folder not found: {folder_path}")
            continue
        
        # Load all image files (jpg, png, jpeg)
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            for image_file in folder_path.glob(ext):
                try:
                    image = cv2.imread(str(image_file))
                    if image is not None:
                        images.append(image)
                        labels.append(category_idx)
                except Exception as e:
                    print(f"Error loading {image_file}: {e}")
    
    
    return np.array(images), np.array(labels)

def extract_features_from_dataset(images: np.ndarray) -> np.ndarray:
    """
    Extract basic features from all images.
    
    Args:
        images: Array of images
        
    Returns:
        Feature matrix (22 features per image)
    """
    features_list = []
    
    for i, image in enumerate(images):
        try:
            # Preprocess
            preprocessed, _ = preprocess_image(image)
            
            # Edge detection
            edges = edge_detection(preprocessed)
            
            # Extract basic features (22 features)
            features = extract_all_features(preprocessed, edges)
            features_list.append(features)
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(images)} images - Using 22 basic features")
        
        except Exception as e:
            print(f"Error processing image {i}: {e}")
    
    return np.array(features_list)

def train_model(X: np.ndarray, y: np.ndarray, test_split: float = 0.2):
    """
    Train waste classifier.
    
    Args:
        X: Feature matrix
        y: Labels
        test_split: Fraction for test set
    """
    from sklearn.model_selection import train_test_split
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=42, stratify=y
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train classifier with optimized hyperparameters
    print("Training Optimized Random Forest Classifier...")
    clf = RandomForestClassifier(
        n_estimators=300,           # More trees for better performance
        max_depth=20,               # Reasonable depth to prevent overfitting
        min_samples_split=5,        # Minimum samples to split node
        min_samples_leaf=2,         # Minimum samples in leaf node
        class_weight='balanced',    # Handle imbalanced classes
        random_state=42,
        n_jobs=-1,                  # Use all cores
        criterion='gini'
    )
    clf.fit(X_train_scaled, y_train)
    
    # Get predictions
    y_pred = clf.predict(X_test_scaled)
    y_pred_proba = clf.predict_proba(X_test_scaled)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Model trained successfully!")
    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    
    # Get unique labels and their names
    unique_labels = sorted(set(y))
    label_names = [CATEGORIES[i] for i in unique_labels]
    print(classification_report(y_test, y_pred, labels=unique_labels, target_names=label_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'")
    
    # Save model and scaler
    Path("data").mkdir(exist_ok=True)
    joblib.dump(clf, "data/trained_model.pkl")
    joblib.dump(scaler, "data/scaler.pkl")
    
    print("\n✅ Model and scaler saved!")
    print("- data/trained_model.pkl")
    print("- data/scaler.pkl")

if __name__ == "__main__":
    print("Loading dataset...")
    images, labels = load_dataset()
    
    if len(images) == 0:
        print("❌ No images found in dataset. Please download dataset first.")
        print("Run: python models/download_dataset.py")
        sys.exit(1)
    
    print(f"Loaded {len(images)} images")
    
    print("\nExtracting features...")
    features = extract_features_from_dataset(images)
    
    print("\nTraining model...")
    train_model(features, labels)