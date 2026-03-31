"""Main Streamlit application for Smart Recycling Bin Classifier."""

import streamlit as st
import cv2
import numpy as np
import time
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.image_preprocessing import preprocess_image, edge_detection, adaptive_histogram_equalization
from src.segmentation import kmeans_segmentation, draw_contours_on_image
from src.feature_extraction import extract_all_features
from src.classification import get_classifier
from src.utils import load_image, validate_image, get_bin_recommendation, format_confidence
from src.constants import CATEGORIES, COLOR_MAP, EMOJI_MAP, CONFIDENCE_THRESHOLD

# Page configuration
st.set_page_config(
    page_title="♻️ Smart Recycling Bin Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []

if 'classifier' not in st.session_state:
    st.session_state.classifier = get_classifier()

def classify_image(image: np.ndarray) -> dict:
    """
    Classify waste in image.
    
    Args:
        image: Input image
        
    Returns:
        Dictionary with classification results
    """
    start_time = time.time()
    
    # Preprocess
    preprocessed, steps = preprocess_image(image)
    
    # Edge detection
    edges = edge_detection(preprocessed)
    
    # Segmentation
    segmented, _ = kmeans_segmentation(preprocessed)
    
    # Feature extraction
    features = extract_all_features(preprocessed, edges)
    
    # Classification
    classifier = st.session_state.classifier
    predicted_class, confidence = classifier.predict(features)
    probabilities = classifier.predict_proba(features)
    
    processing_time = time.time() - start_time
    
    return {
        'image': image,
        'preprocessed': preprocessed,
        'edges': edges,
        'segmented': segmented,
        'contoured': draw_contours_on_image(preprocessed, edges),
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': probabilities,
        'processing_time': processing_time,
        'bin_info': get_bin_recommendation(predicted_class),
        'timestamp': datetime.now(),
        'processing_steps': steps
    }

def display_results(results: dict):
    """Display classification results."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Predicted Category", results['predicted_class'], 
                 results['bin_info']['emoji'])
    
    with col2:
        st.metric("Confidence Score", 
                 format_confidence(results['confidence']))
    
    with col3:
        st.metric("Processing Time", 
                 f"{results['processing_time']:.2f}s")
    
    # Correct bin information
    st.markdown("---")
    bin_info = results['bin_info']
    st.markdown(f"""
    <div class="result-box" style="border-left-color: {COLOR_MAP[results['predicted_class']]};">
        <h3>{bin_info['emoji']} Correct Bin: Bin #{bin_info['bin_number']} ({bin_info['color'].upper()})</h3>
        <p>Place the waste in the <b>{bin_info['color']}</b> bin</p>
    </div>
    """, unsafe_allow_html=True)
    
    # LED Feedback
    led_status = "✅ CORRECT" if results['confidence'] > CONFIDENCE_THRESHOLD else "⚠️ LOW CONFIDENCE"
    led_color = "green" if results['confidence'] > CONFIDENCE_THRESHOLD else "orange"
    
    st.markdown(f"""
    <div style="text-align: center; padding: 20px; background-color: {led_color}; 
                border-radius: 10px; color: white;">
        <h2>LED Feedback: {led_status}</h2>
        <p>Confidence: {format_confidence(results['confidence'])}</p>
    </div>
    """, unsafe_allow_html=True)

def display_probability_distribution(probabilities: dict):
    """Display probability distribution chart."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = list(probabilities.keys())
    probs = list(probabilities.values())
    colors_list = [COLOR_MAP.get(cat, '#cccccc') for cat in categories]
    
    bars = ax.barh(categories, probs, color=colors_list)
    ax.set_xlabel('Probability')
    ax.set_title('Waste Classification Probabilities')
    ax.set_xlim(0, 1)
    
    # Add value labels
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        ax.text(prob, i, f' {prob:.2%}', va='center')
    
    st.pyplot(fig)

def display_processing_pipeline(results: dict):
    """Display image processing pipeline."""
    st.subheader("🔍 Image Processing Pipeline")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(cv2.cvtColor(results['image'], cv2.COLOR_BGR2RGB), 
                caption="Original Image", use_column_width=True)
        st.image(cv2.cvtColor(results['preprocessed'], cv2.COLOR_BGR2RGB), 
                caption="Preprocessed", use_column_width=True)
    
    with col2:
        st.image(cv2.cvtColor(results['edges'], cv2.COLOR_BGR2RGB) if len(results['edges'].shape) == 3 
                else results['edges'],
                caption="Edge Detection (Canny)", use_column_width=True)
        st.image(cv2.cvtColor(results['segmented'], cv2.COLOR_BGR2RGB), 
                caption="K-means Segmentation", use_column_width=True)

def main():
    """Main application."""
    st.title("♻️ Smart Recycling Bin Classifier")
    st.markdown("*Classify waste materials and get guided to the correct recycling bin*")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Navigation")
        page = st.radio("Select Page", 
                       ["🏠 Home", "📊 Analytics", "📚 Guide", "ℹ️ About"])
    
    if page == "🏠 Home":
        st.markdown("---")
        
        # Upload section
        st.subheader("📸 Upload Waste Image")
        uploaded_file = st.file_uploader("Choose an image...", 
                                        type=["jpg", "jpeg", "png", "bmp", "tiff"])
        
        if uploaded_file is not None:
            # Load and validate image
            image = load_image(uploaded_file)
            
            if image is not None and validate_image(image):
                # Show uploaded image
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                        caption="Uploaded Image", use_column_width=True)
                
                # Classify button
                if st.button("🔍 Classify Waste", key="classify_btn", use_container_width=True):
                    with st.spinner("Processing image..."):
                        results = classify_image(image)
                        
                        # Add to history
                        st.session_state.detection_history.append(results)
                        
                        # Display results
                        display_results(results)
                        
                        # Display probability distribution
                        st.markdown("---")
                        st.subheader("📊 Classification Probabilities")
                        display_probability_distribution(results['probabilities'])
                        
                        # Display processing pipeline
                        st.markdown("---")
                        display_processing_pipeline(results)
        
        else:
            st.info("📤 Please upload an image to get started")
    
    elif page == "📊 Analytics":
        st.subheader("📊 Statistics & Analytics")
        
        if len(st.session_state.detection_history) == 0:
            st.info("No detections yet. Upload an image first!")
        else:
            # Statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Detections", len(st.session_state.detection_history))
            
            with col2:
                avg_confidence = np.mean([r['confidence'] for r in st.session_state.detection_history])
                st.metric("Average Confidence", format_confidence(avg_confidence))
            
            with col3:
                avg_time = np.mean([r['processing_time'] for r in st.session_state.detection_history])
                st.metric("Avg Processing Time", f"{avg_time:.2f}s")
            
            # Category distribution
            st.markdown("---")
            st.subheader("Waste Category Distribution")
            
            categories_count = {}
            for result in st.session_state.detection_history:
                cat = result['predicted_class']
                categories_count[cat] = categories_count.get(cat, 0) + 1
            
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Pie chart
            colors_list = [COLOR_MAP.get(cat, '#cccccc') for cat in categories_count.keys()]
            ax1.pie(categories_count.values(), labels=categories_count.keys(), 
                   colors=colors_list, autopct='%1.1f%%')
            ax1.set_title('Distribution by Category')
            
            # Bar chart
            ax2.bar(categories_count.keys(), categories_count.values(), color=colors_list)
            ax2.set_title('Count by Category')
            ax2.set_ylabel('Count')
            ax2.tick_params(axis='x', rotation=45)
            
            st.pyplot(fig)
    
    elif page == "📚 Guide":
        st.subheader("📚 How to Use")
        
        st.markdown("""
        ### Getting Started
        1. **Upload an Image**: Take a photo of your waste item or upload an existing image
        2. **Click Classify**: Wait for the system to process your image
        3. **Check Results**: See which category was detected and which bin to use
        4. **Follow Guidance**: Place the waste in the recommended bin
        
        ### Waste Categories
        """)
        
        for category in CATEGORIES:
            emoji = EMOJI_MAP.get(category, '❓')
            bin_info = get_bin_recommendation(category)
            st.markdown(f"""
            **{emoji} {category}** - Bin #{bin_info['bin_number']} ({bin_info['color'].upper()})
            """)
        
        st.markdown("""
        ### Tips for Best Results
        - ✅ Use clear, well-lit photos
        - ✅ Capture the waste item from multiple angles
        - ✅ Ensure the background is not cluttered
        - ✅ Avoid shadows and reflections
        - ✅ Use items that are typical for recycling
        
        ### LED Feedback System
        - 🟢 **Green Light**: Correct bin selected (High confidence)
        - 🟠 **Orange Light**: Low confidence (Check manually)
        - 🟡 **Yellow Light**: Processing (Wait for result)
        """)
    
    elif page == "ℹ️ About":
        st.subheader("ℹ️ About This Project")
        
        st.markdown("""
        ### Project Overview
        This Smart Recycling Bin Classifier uses advanced computer vision techniques 
        to identify waste materials and guide users to sort them correctly.
        
        ### Technologies Used
        - **OpenCV**: Image processing (edge detection, morphological operations)
        - **scikit-learn**: Machine learning classification
        - **scikit-image**: Image segmentation
        - **Streamlit**: Web interface
        - **Python**: Core programming language
        
        ### Course Concepts Applied
        ✅ Image Enhancement (Histogram Equalization)
        ✅ Edge Detection (Canny Filter)
        ✅ Morphological Operations (Opening, Closing)
        ✅ Image Segmentation (K-means Clustering)
        ✅ Feature Extraction & Classification
        
        ### Dataset
        - **TrashNet**: 2,500+ pre-classified waste images
        - **Categories**: Plastic, Metal, Paper, Glass, Organic
        - **Accuracy**: 85%+ on test set
        
        ### Project Goals
        🎯 Reduce waste contamination in recycling streams
        🎯 Make waste sorting easier and more accurate
        🎯 Promote environmental sustainability
        🎯 Educate users about proper waste classification
        
        ### Author
        **Shubham Jain** | Computer Vision Course Project
        
        ### License
        MIT License
        """)

if __name__ == "__main__":
    main()