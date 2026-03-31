"""Classification module for waste detection."""

import joblib
import numpy as np
from pathlib import Path
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class WasteClassifier:
    """Classifier for waste materials."""
    
    def __init__(self, model_path: str = None, scaler_path: str = None):
        """
        Initialize classifier.
        
        Args:
            model_path: Path to trained model
            scaler_path: Path to feature scaler
        """
        self.model_path = model_path or "data/trained_model.pkl"
        self.scaler_path = scaler_path or "data/scaler.pkl"
        
        self.model = None
        self.scaler = None
        self.classes = None
        
        self.load_model()
        self.load_scaler()
    
    def load_model(self):
        """Load pre-trained model and scaler."""
        try:
            if Path(self.model_path).exists():
                self.model = joblib.load(self.model_path)
                logger.info("Model loaded successfully")
            else:
                logger.warning(f"Model not found at {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
    
    def load_scaler(self):
        """Load feature scaler."""
        try:
            if Path(self.scaler_path).exists():
                self.scaler = joblib.load(self.scaler_path)
                logger.info("Scaler loaded successfully")
        except Exception as e:
            logger.error(f"Error loading scaler: {str(e)}")
    
    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Predict waste category.
        
        Args:
            features: Feature vector
            
        Returns:
            Tuple of (predicted_class, confidence_score)
        """
        if self.model is None:
            logger.error("Model not loaded")
            return "Unknown", 0.0
        
        try:
            # Reshape features if needed (convert 1D to 2D)
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            # Scale features if scaler available
            if self.scaler is not None:
                features = self.scaler.transform(features)
            
            # Predict
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            confidence = max(probabilities)
            
            from src.constants import CATEGORIES
            predicted_class = CATEGORIES[prediction]
            
            return predicted_class, float(confidence)
        
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return "Unknown", 0.0
    
    def predict_with_probabilities(self, features: np.ndarray) -> dict:
        """
        Get predictions with all class probabilities.
        
        Args:
            features: Feature vector
            
        Returns:
            Dictionary with predictions and probabilities
        """
        if self.model is None:
            return {}
        
        try:
            # Reshape features if needed (convert 1D to 2D)
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            if self.scaler is not None:
                features = self.scaler.transform(features)
            
            probabilities = self.model.predict_proba(features)[0]
            
            from src.constants import CATEGORIES
            
            result = {}
            for i, category in enumerate(CATEGORIES):
                result[category] = float(probabilities[i])
            
            return result
        
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return {}
    
    def predict_proba(self, features: np.ndarray) -> dict:
        """
        Get class probabilities (sklearn-compatible method).
        
        Args:
            features: Feature vector
            
        Returns:
            Dictionary with category probabilities
        """
        return self.predict_with_probabilities(features)

# Global classifier instance
_classifier = None

def get_classifier() -> WasteClassifier:
    """Get or create global classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = WasteClassifier()
    return _classifier