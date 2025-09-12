# Facial Emotion Recognition using CNN with PyTorch

This project implements a real-time facial emotion recognition system using a Convolutional Neural Network (CNN) architecture. The model is trained on the FER2013 dataset and deployed for live emotion detection through a PC camera with a graphical user interface.

## Project Structure

### 1. CNN Architecture (`cnn.py`)
- Contains the custom CNN model architecture for emotion classification
- Implements convolutional layers, pooling layers, and fully connected layers
- Designed to process facial images and output emotion predictions

### 2. Data Preprocessing (`fer_data.py`)
- Defines a custom dataset class `CustomDatasetLoader` built upon PyTorch's `ImageFolder`
- Implements data augmentation pipeline for training images
- Creates a custom weighted sampler to handle class imbalance in the dataset
- Splits the full dataset into training, validation, and test subsets

### 3. Model Training (`train.py`)
- Defines the model, loss criterion, optimizer, and learning rate scheduler
- Implements the training loop with validation after each epoch
- Includes learning rate reduction based on validation performance
- Saves best and final model checkpoints
- Generates training/validation accuracy and loss plots
- Produces classification reports and confusion matrices for evaluation
- Creates timestamped directories for organizing training results

### 4. Real-time Application (`main.py`)
- Loads the trained model parameters for inference
- Implements live camera feed capture using PC camera
- Creates a GUI interface to display emotion predictions in real-time
- Processes video frames through the model and displays results with bounding boxes

## Dataset
The model is trained on the FER2013 dataset, which contains 35,887 grayscale images of faces categorized into seven emotions:
- Anger
- Disgust
- Fear
- Happiness
- Sadness
- Surprise
- Neutral

## Usage
1. Preprocess the data: `python fer_data.py`
2. Train the model: `python train.py`
3. Run real-time detection: `python main.py`

## Requirements
- Python 3.x
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

## Results
The system provides real-time emotion detection with visual feedback showing the predicted emotion class and confidence level. Training results include accuracy/loss curves and performance metrics on the test set.
