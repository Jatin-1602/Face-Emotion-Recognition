# Face-Emotion-Recognition

## Introduction

The model is trained on the FER-2013 dataset which is available on Kaggle, and is used to classify the emotion on a person's face into one of seven categories, using deep convolutional neural networks.
FER-2013 dataset consists of 35887 grayscale, 48x48 sized face images with seven emotions - angry, disgusted, fearful, happy, neutral, sad and surprised.

Dataset Link : https://www.kaggle.com/datasets/deadskull7/fer2013


## TechStack
* Python 3
* OpenCV
* Tensorflow

## Real-Time Emotion Detection Algorithm
* First, the haar cascade method is used to detect faces in each frame of the webcam feed.
* The region of image containing the face is resized to 48x48 and is passed as input to the CNN.
* The network outputs a list of softmax scores for the seven classes of emotions.
* The emotion with maximum score is displayed on the screen.


## Additonal Info
`Prep_Images.py` is used for generating images from csv file for simplicity and understandability.<br />
`Model.py` : Training of model has been done. <br />
`Webcam.py` contains code for real time emotion detection of single/multiple person(s) using webcam feed. <br />
`model.h5` : Trained Model with (Training Accuracy : 93% and Validation Accuracy : 61%)
