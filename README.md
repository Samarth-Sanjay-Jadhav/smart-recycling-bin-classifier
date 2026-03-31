# ♻️ Smart Recycling Bin Classifier

A machine learning-powered waste classification system that uses computer vision to automatically detect and categorize recyclable materials. The project uses an advanced image processing pipeline combined with a Random Forest classifier to achieve reliable accuracy in waste sorting.

---

## 👤 Author

- **Name**: Samarth Sanjay Jadhav
- **GitHub**: [Samarth-Sanjay-Jadhav](https://github.com/Samarth-Sanjay-Jadhav)
- **Course**: Computer Vision

---

## 🎯 Project Overview

The Smart Recycling Bin Classifier automatically identifies waste materials in images and classifies them into one of 5 categories:

| Category      | Examples                              |
|---------------|---------------------------------------|
| 🧴 Plastic    | PET bottles, containers, bags         |
| 🥫 Metal      | Aluminum cans, steel, metal containers|
| 📰 Paper      | Newspapers, magazines, sheets         |
| 📦 Cardboard  | Boxes, packaging materials            |
| 🫙 Glass      | Glass bottles, jars, containers       |

**Current Model Accuracy: 73.01%**

---

## 🚀 Features

- **Real-time Classification** — Upload images or use webcam for live waste detection
- **High Accuracy** — 73.01% accuracy trained on 2,390+ images
- **Advanced CV Pipeline** — Histogram equalization, Canny edge detection, K-means segmentation
- **22-Feature Extraction** — Color, texture, and shape-based features
- **Confidence Scoring** — Prediction confidence displayed for each classification
- **Interactive UI** — Streamlit-based web interface
- **Comprehensive Visualization** — View preprocessing steps and classification probabilities

---

## 💻 System Requirements

- Python 3.8+
- Windows / macOS / Linux
- Minimum 2GB RAM
- Webcam *(optional, for real-time classification)*

---

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Samarth-Sanjay-Jadhav/smart-recycling-bin-classifier.git
cd smart-recycling-bin-classifier
```

### 2. Create a Virtual Environment
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

---

## 🎮 Running the Application

```bash
streamlit run app_streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

### How to Use
1. **Upload Image** — Click "Browse files" and select a waste image
2. **Use Webcam** — Click the camera icon for real-time classification
3. **View Results** — See predicted category and confidence score
4. **Analyze** — Review preprocessing steps and probability distribution

---

## 📁 Project Structure

```
smart-recycling-bin-classifier/
├── app_streamlit_app.py           # Main Streamlit application
├── models_train_classifier.py     # Model training script
├── config.yaml                    # Configuration file
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── .gitignore                     # Git ignore rules
├── src/
│   ├── __init__.py
│   ├── classification.py          # Classifier interface
│   ├── constants.py               # Category constants and mappings
│   ├── feature_extraction.py      # 22-feature extraction pipeline
│   ├── image_preprocessing.py     # Preprocessing (histogram eq, CLAHE)
│   ├── segmentation.py            # K-means segmentation
│   └── utils.py                   # Utility functions
├── data/
│   ├── dataset/                   # Training dataset (2,390 images)
│   │   ├── plastic/               (482 images)
│   │   ├── metal/                 (410 images)
│   │   ├── paper/                 (594 images)
│   │   ├── cardboard/             (403 images)
│   │   └── glass/                 (501 images)
│   ├── trained_model.pkl          # Trained Random Forest model
│   └── scaler.pkl                 # StandardScaler for features
```

---

## 📊 Model Training

To train the model from scratch:
```bash
python models_train_classifier.py
```

This will:
- Load all images from `data/dataset/`
- Extract 22 features per image
- Train a Random Forest classifier (300 trees)
- Save the model to `data/trained_model.pkl`
- Save the scaler to `data/scaler.pkl`
- Generate a confusion matrix visualization

---

## 🧠 Computer Vision Pipeline

### 1. Image Preprocessing
- Convert BGR → LAB color space
- Histogram equalization for contrast enhancement
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Gaussian blur (5×5 kernel)
- Morphological operations (open/close with elliptical kernel)

### 2. Edge Detection
- Canny edge detection with thresholds (100, 200)
- Identifies material boundaries and structural features

### 3. Segmentation
- K-means clustering (k=3)
- Watershed algorithm
- Contour extraction and analysis

### 4. Feature Extraction (22 Features)
- **Color Features (14)** — HSV channel statistics, color histogram bins
- **Texture Features (4)** — Laplacian edge stats, gradient information
- **Shape Features (4)** — Area, perimeter, circularity, ellipse axes

### 5. Classification
- **Algorithm**: Random Forest Classifier
- **Trees**: 300 | **Max Depth**: 20 | **Min Samples Split**: 5
- **Class Weight**: Balanced
- **Feature Scaling**: StandardScaler normalization

---

## 📈 Performance Metrics

**Overall Accuracy: 73.01%**

| Category  | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Plastic   | 0.77      | 0.72   | 0.74     | 96      |
| Metal     | 0.76      | 0.67   | 0.71     | 82      |
| Paper     | 0.76      | 0.84   | 0.80     | 119     |
| Cardboard | 0.83      | 0.59   | 0.69     | 81      |
| Glass     | 0.61      | 0.77   | 0.68     | 100     |

---

## 🔧 Configuration

Edit `config.yaml` to customize behavior:
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

---

## 📦 Dependencies

- **OpenCV** — Image processing
- **scikit-learn** — Machine learning and classification
- **scikit-image** — Advanced image processing
- **NumPy** — Numerical operations
- **Pandas** — Data manipulation
- **Streamlit** — Web UI framework
- **Pillow** — Image manipulation
- **Matplotlib / Seaborn** — Visualization

See `requirements.txt` for exact versions.

---

## 🔮 Future Improvements

- [ ] Deep learning models (CNN, ResNet)
- [ ] Real-time video stream processing
- [ ] Mobile app deployment
- [ ] Data augmentation for underrepresented categories
- [ ] Multi-object detection in a single image
- [ ] Cloud deployment (AWS / Azure)
- [ ] Docker containerization

---

## 🚨 Troubleshooting

**Model not found**
```bash
python models_train_classifier.py
```

**Port already in use**
```bash
streamlit run app_streamlit_app.py --server.port=8502
```

**OpenCV import error**
```bash
pip install --upgrade opencv-python
```

**Low classification accuracy**
- Use clearer, well-lit images
- Ensure the background is not cluttered
- Retrain the model with more data

---

## 🙏 Acknowledgments

- [TrashNet Dataset](https://github.com/garythung/trashnet) by Gary Thung
- OpenCV Community
- Streamlit Framework
- scikit-learn Library

---

## 📚 References

- OpenCV Docs: https://docs.opencv.org/
- scikit-learn: https://scikit-learn.org/
- Streamlit: https://streamlit.io/

---

**Happy Recycling! ♻️**
