# Eye Cataract Detection System

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

## 🎯 Project Overview

A comprehensive machine learning system for automated cataract detection from eye images using classical ML techniques. This project demonstrates the complete pipeline from image preprocessing to classification, achieving **~99% accuracy** through PCA-based feature extraction and SVM classification, with additional clustering analysis for pattern discovery.

## 🏥 Problem Statement

Cataracts are one of the leading causes of vision loss worldwide. Early detection is crucial for timely treatment and preventing vision impairment. This project automates the screening process using machine learning, providing fast and accurate cataract detection to assist medical professionals.

## 📊 Dataset Information

**Source:** Medical eye images dataset stored in Google Drive  
**Structure:**
```
train/
├── cataract/        # Images with cataract
├── normal/          # Healthy eye images
└── mature/          # Mature cataract cases
```

**Specifications:**
- **Format:** JPG images
- **Classes:** Binary classification (Cataract / Normal)
- **Distribution:** Balanced dataset
- **Processed Size:** 128×128 pixels (grayscale)
- **Split Ratio:** 70% Train, 15% Validation, 15% Test

## 🚀 Solution Pipeline

### 1. Image Preprocessing
- Resizing to 128×128 pixels
- Grayscale conversion
- Gaussian blur (5×5 kernel) for noise reduction
- Pixel normalization with StandardScaler
- Feature vector flattening (16,384 features)

### 2. Dimensionality Reduction (PCA)
- **Variance Retained:** 95%
- **Original Features:** 16,384 (128×128)
- **Reduced Features:** ~50 components
- **Training Time Improvement:** 75% faster
- **Purpose:** Remove noise, prevent overfitting, improve efficiency

### 3. Classification (SVM)
**Hyperparameter Tuning via GridSearchCV:**
- **C:** [0.1, 1, 10, 100]
- **gamma:** [0.001, 0.01, 0.1]
- **kernel:** ['rbf', 'linear']
- **Cross-validation:** 5-fold

**Models Trained:**
- RBF Kernel SVM (optimized via grid search)
- Linear Kernel SVM (baseline comparison)

### 4. Clustering Analysis (K-Means)
- **Optimal K:** 3 clusters (determined by elbow method)
- **Feature Space:** PCA-reduced features
- **Purpose:** Discover natural groupings without labels
- **Visualizations:** Scatter plots, cluster distribution, sample images

## 📈 Results

| Metric | Score |
|--------|-------|
| **Validation Accuracy** | ~99% |
| **Test Accuracy** | ~99% |
| **Precision** | 98.8% |
| **Recall** | 99.2% |
| **F1-Score** | 99.0% |

**Model Performance:**
- RBF SVM achieved highest accuracy through hyperparameter optimization
- Linear SVM provided comparable results with faster training
- 5-fold cross-validation ensured model generalization
- Separate validation and test sets prevented overfitting

**Confusion Matrix:** High true positive and true negative rates with minimal false classifications

## 🛠️ Technologies Used

- **Language:** Python 3.x
- **ML Libraries:** scikit-learn (SVM, PCA, K-Means, GridSearchCV)
- **Image Processing:** OpenCV
- **Data Manipulation:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Environment:** Google Colab

## 📁 Repository Structure
```
Cataract-Detection-ML/
├── ds_project_eye_cataract.ipynb    # Main Jupyter notebook
├── requirements.txt                  # Dependencies
└── README.md                         # Documentation
```

## 🔧 Installation

### Prerequisites
```bash
pip install -r requirements.txt
```

### Required Libraries
```
numpy>=1.21.0
pandas>=1.3.0
opencv-python>=4.5.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 💻 Usage

### Running in Google Colab

1. **Mount Google Drive:**
```python
from google.colab import drive
drive.mount('/content/drive')
```

2. **Set Dataset Path:**
```python
train_dir = "/content/drive/MyDrive/DS CATARACT /train"
```

3. **Run Complete Pipeline:**
```python
# Load and preprocess images
X, y, original_images = load_images(train_dir)

# Train-test split and scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train_scaled = scaler.fit_transform(X_train)

# Apply PCA
X_train_pca = pca.fit_transform(X_train_scaled)

# Train SVM
svm_model = grid.best_estimator_
svm_model.fit(X_train_pca, y_train)
```

4. **Make Predictions:**
```python
classify_new_image("/path/to/test/image.jpg")
```

## 🎓 Key Techniques Demonstrated

- **Image Preprocessing:** Normalization, grayscale conversion, noise reduction
- **Dimensionality Reduction:** PCA for feature extraction and computational efficiency
- **Supervised Learning:** SVM classification with hyperparameter optimization
- **Unsupervised Learning:** K-Means clustering for pattern discovery
- **Model Evaluation:** Accuracy, precision, recall, F1-score, confusion matrix
- **Cross-Validation:** 5-fold CV for robust model selection
- **Visualization:** Data exploration, cluster analysis, result interpretation

## 🔍 Model Insights

The model successfully distinguishes between cataract and normal eye images by:
1. Extracting essential visual features through PCA
2. Learning optimal decision boundaries via SVM
3. Achieving high accuracy on balanced test data
4. Identifying natural patterns through unsupervised clustering

The 99% accuracy demonstrates strong generalization capability with proper preprocessing and feature engineering.

## 📊 Visualizations Included

- Original vs. Preprocessed images comparison
- Elbow method curve for optimal K selection
- PCA scatter plot showing cluster separation
- Confusion matrix heatmap
- Cluster distribution (pie chart and bar plot)
- Sample images from each cluster
- Classification results on test images

## 💼 Author

**Jasmine D**  
Data Analyst | SQL | Power BI | Python  
[LinkedIn](https://linkedin.com/in/djasmine1610) | [Email](mailto:djasmine1610@gmail.com) | [GitHub](https://github.com/djasmine123)

## 📝 License

This project is for educational and portfolio demonstration purposes. The dataset used is publicly available for academic research.

## 🙏 Acknowledgments

This project was developed as part of Data Science coursework, demonstrating the practical application of classical machine learning techniques to medical image classification. Special thanks to the open-source community for providing tools and datasets.

---

⭐ If you found this project helpful, please consider giving it a star!
