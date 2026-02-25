# Eye Cataract Detection System

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)

## 🎯 Project Overview

Machine Learning-based image classification system for automated cataract detection from medical images, achieving **99% accuracy** (98.8% precision, 99.2% recall) using Support Vector Machines (SVM).

## 🏥 Problem Statement

Early detection of cataracts is crucial for preventing vision loss. This project automates the detection process using ML to assist medical professionals in diagnosis.

## 🚀 Solution

### Pipeline

1. **Image Preprocessing**
   - Loaded medical images using OpenCV
   - Normalized pixel values
   - Resized to consistent dimensions

2. **Feature Extraction**
   - Applied Principal Component Analysis (PCA)
   - Reduced dimensionality from 1024 to 50 features
   - Retained 95% of variance

3. **Clustering**
   - K-Means clustering for pattern identification
   - Optimal K determined using elbow method

4. **Classification**
   - Support Vector Machine (SVM) with RBF kernel
   - Trained on balanced medical image dataset
   - Achieved 99% test accuracy

## 📊 Results

| Metric | Score |
|--------|-------|
| **Accuracy** | 99.0% |
| **Precision** | 98.8% |
| **Recall** | 99.2% |
| **F1-Score** | 99.0% |
| **Training Time** | 75% faster (after PCA) |

## 🛠️ Technologies

- **Language:** Python 3.x
- **ML Library:** scikit-learn
- **Deep Learning:** TensorFlow
- **Image Processing:** OpenCV
- **Data Manipulation:** pandas, numpy
- **Visualization:** matplotlib, seaborn

## 🎓 Key Techniques Demonstrated

- Image preprocessing and normalization
- Dimensionality reduction using PCA
- K-Means clustering for pattern discovery
- Support Vector Machine classification
- Model evaluation and performance metrics

## 💼 Author

**Jasmine D**  
Data Analyst | SQL | Power BI | Python  
[LinkedIn](https://linkedin.com/in/djasmine1610) | [Email](mailto:djasmine1610@gmail.com)

## 📝 License

This project is for portfolio demonstration purposes.

---

⭐ If you found this project helpful, please consider giving it a star!
