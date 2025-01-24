# Heart-Prediction-model
Heart Disease Prediction Using PyTorch
This repository contains a deep learning project for predicting the presence of heart disease based on patient data. The model is implemented using PyTorch and trained on a preprocessed dataset to achieve high accuracy in classification.

Features
Preprocessing of raw data (handling missing values, scaling features, and encoding categorical variables).
Binary classification of heart disease presence.
Implementation of a feed-forward neural network using PyTorch.
Training, validation, and testing workflows with performance evaluation metrics.
Visualization of training/testing loss and accuracy trends over epochs.
Dataset
The dataset used is the Heart Disease Prediction Dataset, which includes features like age, sex, chest pain type, resting blood pressure, cholesterol levels, and other patient information. The target variable indicates the presence of heart disease (1) or absence (0).

Highlights
Model Architecture:
Input Layer: Features from the dataset.
Hidden Layers: 2 layers with ReLU activation.
Output Layer: Sigmoid activation for binary classification.
Training:
Optimizer: Adam.
Loss Function: Binary Cross-Entropy Loss.
Regularization: Dropout layers to prevent overfitting.
Performance:
Achieved a high test accuracy of 86.89%.
Visualization of loss and accuracy over epochs.
Visualization
The project includes graphs to illustrate the training and testing process:

Training vs. Testing Loss: Monitors model performance and overfitting.
Training vs. Testing Accuracy: Tracks improvements in prediction accuracy.
