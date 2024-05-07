# Digit Recognizer using Convolutional Neural Networks (CNNs)

## Overview
This project implements Convolutional Neural Networks (CNNs) to recognize handwritten digits from images. The primary objective is to build and train models capable of accurately classifying digits (0-9) based on input images.

## Introduction
Handwritten digit recognition is a fundamental problem in computer vision and machine learning. It finds applications in various domains, including optical character recognition (OCR), digitizing documents, and payment verification. This project aims to leverage CNNs, a class of deep neural networks well-suited for image recognition tasks, to accurately identify handwritten digits from input images.

## Dataset
The dataset used for this project is the MNIST dataset, a widely-used benchmark dataset for handwritten digit recognition. It consists of 28x28 grayscale images of handwritten digits (0-9) along with their corresponding labels. The dataset is split into training and testing sets, with the training set used for model training and the testing set for evaluation.

## Approach
1. **Data Preprocessing**: The input images undergo preprocessing steps such as normalization and resizing to prepare them for model training.
2. **Model Architecture**: We design and implement CNN architectures suitable for digit recognition tasks. These architectures typically consist of convolutional layers, pooling layers, and fully connected layers.
3. **Model Training**: The CNN models are trained on the training dataset using various neural networks. We monitor the training process to prevent overfitting and ensure model generalization.
4. **Evaluation**: The trained models are evaluated on the testing dataset to assess their performance in digit recognition. Evaluation metrics such as accuracy, precision, recall, and F1-score are computed to measure model effectiveness.

## Results
The CNN models achieve impressive results in recognizing handwritten digits from input images. High accuracy scores demonstrate the effectiveness of the CNN architectures and training process in accurately classifying digits.

## Link
View the Jupyter notebook [here](https://github.com/mikemiller97/digit-recognizer/blob/main/digit-recognizer-computer-vision.ipynb) on GitHub to explore the project code and view my results.
