# Real-Time-Emotion-Detection-Using-CNN



---

## Overview

This project implements a real-time emotion detection system using Convolutional Neural Networks (CNN). It utilizes deep learning techniques to recognize facial expressions in real-time through a computer's webcam.

## Dataset

The project utilizes the Face Expression Recognition Dataset available on Kaggle. This dataset contains grayscale images of size 48x48 pixels, each labeled with one of seven different emotions (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise).

**Dataset Link:** [Face Expression Recognition Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)

The dataset is structured as follows:

- `train` directory: Contains training images divided into seven subdirectories, each representing a different emotion class.
- `validation` directory: Contains validation images similarly divided into seven subdirectories based on emotion classes.

## Prerequisites

Ensure you have the following libraries installed:

- `matplotlib`
- `numpy`
- `pandas`
- `seaborn`
- `keras`

## Model Architecture

The CNN model architecture comprises several convolutional and pooling layers, followed by fully connected layers for classification. Here's a summary of the model architecture:

- Convolutional layers with batch normalization, ReLU activation, and dropout
- Fully connected layers with batch normalization and ReLU activation
- Output layer with softmax activation for multi-class classification

## Training

The model is trained using the Adam optimizer with a learning rate of 0.001 and categorical cross-entropy loss. Training is performed for a specified number of epochs with early stopping and learning rate reduction to prevent overfitting.

## Evaluation

The model's performance is evaluated on both training and validation datasets in terms of loss and accuracy. Plots are generated to visualize the training and validation metrics.

## Real-Time Emotion Detection

The real-time emotion detection utilizes OpenCV for face detection using the Haar cascade classifier. Once a face is detected, the CNN model predicts the emotion label in real-time. Emotion labels are displayed on the video feed captured by the webcam.

## Usage

To use the real-time emotion detection system:

1. Run the provided Python script.
2. Ensure your webcam is connected and functioning properly.
3. The script will display the webcam feed with emotion labels overlaid on detected faces.
4. Press 'q' to quit the application.

## Files

- `model.h5`: Trained CNN model saved in HDF5 format.
- `haarcascade_frontalface_default.xml`: Pre-trained Haar cascade classifier for face detection.

## References

- [Face Expression Recognition Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)
- [OpenCV Documentation](https://opencv.org/)
- [Keras Documentation](https://keras.io/)

---

