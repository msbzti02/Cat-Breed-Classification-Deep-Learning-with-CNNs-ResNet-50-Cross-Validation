##  Cat Breed Classification — Deep Learning with CNNs, ResNet-50 & Cross-Validation
###  Overview

This project implements a multi-class image classification system to identify five cat breeds:
Bengal, Domestic Shorthair, Maine Coon, Ragdoll, Siamese.

It combines custom convolutional networks and transfer learning (ResNet-50), using k-fold cross-validation, grid search, and data augmentation to optimize model generalization.

The core focus is on deep learning experimentation methodology — model selection, hyperparameter tuning, cross-validation, and evaluating augmentation effects on performance.

--- 
###  Machine Learning Focus

| **Field**              | **Techniques Applied**                                        |
|-------------------------|---------------------------------------------------------------|
| **Deep Learning**       | CNNs, Transfer Learning, Fine-tuning                          |
| **Model Optimization**  | Grid Search, k-Fold Cross Validation                          |
| **Regularization**      | Dropout, Batch Normalization                                  |
| **Data Augmentation**   | Flips, Color Jitter, Brightness, Noise                        |
| **Metrics**             | Accuracy, Top-3 Accuracy, Precision, Recall, F1-Score         |
| **Evaluation Tools**    | Confusion Matrix, Misclassification Analysis    

---

##  Deep Learning Methods & Architecture
### 1️ SimpleCNN — Custom Convolutional Model

Built from scratch to establish a baseline.

###  Model Architecture
**3 Convolutional Blocks**, each composed of:
  - `Conv2d`
  - `BatchNorm2d`
  - `ReLU`
  - `MaxPool2d`

### Fully Connected Classifier
- `Flatten`
- `Linear(512)`
- `ReLU`
- `Dropout(0.5)`
- `Linear(num_classes)`
- `LogSoftmax`
### Optimizer
- `Adam`
### Loss
- `Negative Log Likelihood Loss (NLLLoss)`

Purpose:
Establish baseline accuracy using a custom lightweight CNN with ~3M parameters.

---


##  ResNet-50 — Transfer Learning

### Pretrained Model
- **Base Architecture:** `ResNet-50`
- **Pretrained Weights:** `ImageNet`
- **Frozen Backbone:** Early convolutional layers frozen to preserve low-level feature representations
### Modified Classifier
- `Linear(in_features, num_classes)`
- `LogSoftmax(dim=1)`
### Training Setup
- Fine-tuned with **small learning rates**
- Regularized using **5-Fold Cross Validation** to reduce overfitting
- Optimized for **fast convergence** and **strong generalization** on limited data
**Goal:**  
Leverage pretrained ImageNet features to accelerate training and improve performance when data is scarce.

---
##  ResNet-50 + Data Augmentation

### Augmentation Pipeline
Designed to enhance dataset diversity and mitigate overfitting through real-time image transformations.

**Applied Techniques:**
- `HorizontalFlip` / `VerticalFlip`
- `Brightness` and `Contrast` Jitter
- `RandomRotation`
- `Additive Gaussian Noise`

### Impact
- Improved model **generalization**
- Increased **test accuracy**, particularly **Top-3 performance**

**Goal:**  
Boost robustness and reduce overfitting by exposing the model to a broader range of visual variations.

---

##  Cross-Validation Setup

### Configuration
- **k = 5 (StratifiedKFold)**
- Maintains **balanced class distribution** across all folds
- Measures **performance consistency** and **model robustness**
- Logs **per-epoch metrics** including training and validation loss/accuracy
- Identifies the **optimal hyperparameter combination** based on lowest validation loss

### Output
- Mean **accuracy** and **loss curves** across folds  
- Best-performing **parameter set** for each model  
- Validation **stability metrics** reflecting model reliability

**Goal:**  
Achieve stable, high-performing configurations through systematic evaluation across multiple data splits.

--- 
##  Evaluation Metrics

| **Metric** | **Description** |
|-------------|-----------------|
| **Accuracy** | Overall correct classifications |
| **Top-3 Accuracy** | Measures if the true label appears within the top 3 predictions |
| **Precision / Recall / F1-Score** | Evaluates per-class classification performance |
| **Confusion Matrix** | Visualizes misclassification patterns across classes |
| **Cross-Validation Curves** | Tracks validation loss and accuracy across folds |
<img width="398" height="278" alt="image" src="https://github.com/user-attachments/assets/c0657da3-61b6-42b2-acb1-394bf381e814" />

<img width="381" height="286" alt="image" src="https://github.com/user-attachments/assets/50b6d947-b9dd-414b-bcc3-863520a99904" />
<img width="650" height="491" alt="image" src="https://github.com/user-attachments/assets/abd492c1-0ace-4310-a1b4-1675dfd8fba4" />

<img width="387" height="267" alt="image" src="https://github.com/user-attachments/assets/383c9a94-5f02-4d2b-8c51-9610d0e4438e" />

---
##  Experimental Summary

| **Model** | **Training Method** | **Data** | **Best Accuracy** | **F1-Score** | **Top-3 Accuracy** |
|------------|--------------------|----------|------------------:|--------------:|-------------------:|
| **SimpleCNN** | Baseline CNN | Original | ~87% | ~85% | ~93% |
| **ResNet-50** | Transfer Learning | Original | ~94% | ~92% | ~97% |
| **ResNet-50 + Augmentation** | Transfer + Augmentation | Augmented | **96%+** | **94%+** | **99%** |

---
🧑‍💻 Author

`Mourad Sleem`

`Deep Learning Researcher & ML Engineer`

📧 moradbshina@gmail.com
