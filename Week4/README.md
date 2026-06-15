# Week 4 - CIFAR-10 ANN vs CNN Learning Project

## Overview

This project explores image classification using the CIFAR-10 dataset and compares the performance of Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN). The goal is to understand why CNNs are better suited for image-related tasks by analyzing their architecture, feature extraction capabilities, and classification performance.

---

## Dataset

The CIFAR-10 dataset contains:

- 60,000 color images
- 10 image categories
- 50,000 training images
- 10,000 testing images
- Image size: 32 × 32 × 3

### Classes

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

---

## Topics Covered

### Data Preparation
- Loading CIFAR-10 dataset
- Data visualization
- Image normalization

### ANN Implementation
- Dense Neural Network
- Dropout Regularization
- Model Training and Evaluation

### CNN Implementation
- Convolutional Layers
- Max Pooling
- Batch Normalization
- Feature Extraction

### Model Enhancement
- Improved ANN Architecture
- Data Augmentation
  - Random Flip
  - Random Rotation
  - Random Zoom

### Performance Evaluation
- Accuracy Comparison
- Learning Curve Analysis
- Model Performance Visualization

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib

---

## Results

| Model | Accuracy |
|---------|---------|
| ANN | ~43% |
| Improved ANN | ~45% |
| CNN | ~70% |
| Augmented CNN | ~65% |

### Key Observation

CNN significantly outperformed ANN because convolutional layers can capture spatial patterns and important image features more effectively than fully connected networks.

---

## Learning Outcomes

Through this project, I learned:

- Fundamentals of image classification
- Differences between ANN and CNN architectures
- Importance of feature extraction in computer vision
- Data preprocessing and augmentation techniques
- Model evaluation and performance comparison using deep learning

---

## Conclusion

The project demonstrated that CNNs are far more effective than ANNs for image classification tasks. While ANN achieved moderate performance, CNN was able to learn spatial features directly from images, resulting in significantly higher accuracy. This experiment provided practical experience in deep learning, computer vision, and model optimization techniques.

---

## Author

**Ankita Yadav**  
B.Tech CSE | JECRC Foundation  
Exploring Machine Learning, Deep Learning, and AI through hands-on projects.
