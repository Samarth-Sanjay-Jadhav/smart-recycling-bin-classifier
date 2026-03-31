# ♻️ Smart Recycling Bin Classifier

A machine learning-powered waste classification system that uses computer vision to automatically detect and categorize recyclable materials. This project leverages advanced image processing and Random Forest classification to achieve high accuracy in waste sorting.

## 🎯 Project Overview

The Smart Recycling Bin Classifier automatically identifies waste materials in images and classifies them into one of 5 categories:
- **Plastic** - PET bottles, plastic containers, bags
- **Metal** - Aluminum cans, steel, metal containers
- **Paper** - Cardboard, newspapers, magazines
- **Cardboard** - Boxes, packaging materials
- **Glass** - Glass bottles, jars, containers

**Current Model Accuracy: 73.01%**

## 🚀 Features

✅ **Real-time Classification** - Upload images or use webcam for live waste detection
✅ **High Accuracy** - 73.01% accuracy on 2,390+ training images
✅ **Advanced CV Pipeline** - Histogram equalization, Canny edge detection, K-means segmentation
✅ **22-Feature Extraction** - Color, texture, and shape features
✅ **Confidence Scoring** - See prediction confidence for each classification
✅ **Interactive UI** - Streamlit-based web interface
✅ **Comprehensive Visualization** - View preprocessing steps and classification probabilities  

## � System Requirements

- Python 3.8+
- Windows/macOS/Linux
- Minimum 2GB RAM
- Webcam (optional, for real-time classification)

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/smart-recycling-bin-classifier.git
cd smart-recycling-bin-classifier
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 🎮 Running the Application

### Start the Streamlit App
```bash
streamlit run app_streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

### Usage
1. **Upload Image**: Click "Browse files" and select a waste image
2. **Use Webcam**: Click the camera icon for real-time classification
3. **View Results**: See predicted category and confidence score
4. **Analyze**: Review preprocessing steps and probability distribution

## 📁 Project Structure

```
smart-recycling-bin-classifier/
├── app_streamlit_app.py           # Main Streamlit application
├── models_train_classifier.py      # Model training script
├── config.yaml                     # Configuration file
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
├── src/
│   ├── __init__.py
│   ├── classification.py           # Classifier interface
│   ├── constants.py                # Category constants and mappings
│   ├── feature_extraction.py       # 22-feature extraction pipeline
│   ├── image_preprocessing.py      # Image preprocessing (histogram eq, CLAHE)
│   ├── segmentation.py             # K-means segmentation
│   └── utils.py                    # Utility functions
├── data/
│   ├── dataset/                    # Training dataset (2,390 images)
│   │   ├── plastic/                (482 images)
│   │   ├── metal/                  (410 images)
│   │   ├── paper/                  (594 images)
│   │   ├── cardboard/              (403 images)
│   │   ├── glass/                  (501 images)
│   │   └── trash/                  (Validation only)
│   ├── trained_model.pkl           # Trained Random Forest model
│   └── scaler.pkl                  # StandardScaler for features
└── .git/                           # Git repository data
```

## 📊 Model Training

### Train from Scratch
```bash
python models_train_classifier.py
```

This will:
- Load all images from `data/dataset/`
- Extract 22 features per image
- Train Random Forest classifier (300 trees)
- Save model to `data/trained_model.pkl`
- Save scaler to `data/scaler.pkl`
- Generate confusion matrix visualization

## 🧠 Computer Vision Pipeline

### 1. **Image Preprocessing**
- Convert BGR → LAB color space
- Histogram equalization for contrast enhancement
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Gaussian blur (5×5 kernel)
- Morphological operations (open/close with elliptical kernel)

### 2. **Edge Detection**
- Canny edge detection with thresholds (100, 200)
- Identifies material boundaries

### 3. **Segmentation**
- K-means clustering (k=3)
- Watershed algorithm
- Contour extraction and analysis

### 4. **Feature Extraction** (22 features)
- **Color Features (14)**
  - HSV channel statistics (mean, std per channel)
  - Color histogram bins
- **Texture Features (4)**
  - Laplacian edge statistics (mean, std)
  - Gradient information
- **Shape Features (4)**
  - Area and perimeter
  - Circularity
  - Ellipse major/minor axes

### 5. **Classification**
- **Algorithm**: Random Forest Classifier
- **Trees**: 300
- **Max Depth**: 20
- **Min Samples Split**: 5
- **Class Weight**: Balanced
- **Feature Scaling**: StandardScaler normalization

## 📈 Performance Metrics

### Overall Accuracy: 73.01%

### Per-Category Performance:
| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Plastic  | 0.77      | 0.72   | 0.74     | 96      |
| Metal    | 0.76      | 0.67   | 0.71     | 82      |
| Paper    | 0.76      | 0.84   | 0.80     | 119     |
| Cardboard| 0.83      | 0.59   | 0.69     | 81      |
| Glass    | 0.61      | 0.77   | 0.68     | 100     |

## 🔧 Configuration

Edit `config.yaml` to customize:
```yaml
categories:
  - Plastic
  - Metal
  - Paper
  - Cardboard
  - Glass

confidence_threshold: 0.5
image_size: [640, 480]
```

## 📦 Dependencies

- **OpenCV** (cv2) - Image processing
- **scikit-learn** - Machine learning, classification
- **scikit-image** - Advanced image processing
- **NumPy** - Numerical operations
- **Pandas** - Data manipulation
- **Streamlit** - Web UI framework
- **Pillow** - Image manipulation
- **Matplotlib/Seaborn** - Visualization

See `requirements.txt` for exact versions.

## 🔮 Future Improvements

- [ ] Deep learning models (CNN, ResNet)
- [ ] Real-time video stream processing
- [ ] Mobile app deployment
- [ ] Data augmentation for underrepresented categories
- [ ] Multi-object detection in single image
- [ ] Cloud deployment (AWS/Azure)
- [ ] Docker containerization

## 🚨 Troubleshooting

### Issue: "Model not found"
**Solution**: Download pre-trained model or train new model
```bash
python models/download_dataset.py
python models/train_classifier.py
```

### Issue: "Port already in use"
**Solution**: Specify different port
```bash
streamlit run app/streamlit_app.py --server.port=8502
```

### Issue: "OpenCV import error"
**Solution**: Reinstall opencv-python
```bash
pip install --upgrade opencv-python
```

### Issue: Low classification accuracy
**Solution**: 
- Use clearer, well-lit images
- Ensure backgrounds aren't cluttered
- Retrain model with more data

## 📜 License

MIT License - see LICENSE file for details

## 📧 Contact & Support

- **Author**: Shubham Jain
- **GitHub**: [@shubhamjain1402](https://github.com/shubhamjain1402)

## 🙏 Acknowledgments

- TrashNet Dataset by [Gary Thung](https://github.com/garythung/trashnet)
- OpenCV Community
- Streamlit Framework
- scikit-learn Library

## 📚 References

- OpenCV Documentation: https://docs.opencv.org/
- scikit-learn: https://scikit-learn.org/
- Streamlit: https://streamlit.io/
- Image Processing Papers and Resources

---

**Happy Recycling! ♻️**
