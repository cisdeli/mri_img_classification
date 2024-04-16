# Classifying levels of Alzheimer's disease using MRI and machine learning

The objective of this project is to test different machine learning classification algorithms as well as different 
image processing techniques on the training set. Additionally, we will use the trained models to predict the level of dementia
based on the MRI images.

This project was developed for a Machine Learning class at the University of São Paulo (USP) in 2021. Because of this, most of the reports are in Portuguese.
I will try to translate them to English as soon as possible. For now use this markdown file as a guide to the project.

## Relevancy of the project

Alzheimer's disease is a neurodegenerative disease that affects millions of people worldwide. The disease is known for a really complex diagnosis process, which involves a lot of different exams and specialists.
The early detection of the disease is crucial for the patient's treatment and quality of life. 
The use of machine learning algorithms to classify the level of dementia based on MRI images can help doctors to diagnose the disease more accurately and quickly.

## Dataset Information

* The dataset can be accessed on (Kaggle)[https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection]
* 6400 MRI images of the brain.
* Resolution is 128x128 pixels.
* Grayscale images.
An example of the images can be seen on Figure 1.

![MRI Image example](assets/mri_example.jpeg "MRI Image Example")
*Figure.1 - MRI Image example from the dataset.*

* The dataset is divided into 4 classes: Non-Demented, Very Mild Demented, Mild Demented, Moderate Demented.
The dataset is unbalanced, with the majority of the images being classified as Non-Demented.
To account for this factor we kept the same distribution of classes in the training and test sets, as shown on Figure 2 and Figure 3
--- fig 2 and 3 here---

## Models Tested
* K-Nearest Neighbors *(KNN)*.
* Decision Tree *(DT)*.
* Support Vector Machine *(SVM)*.
* Multilayer Perceptron *(MLP)*.
* Convolutional Neural Network *(CNN)8.

## Image Processing

First, it was applied different levels of Gaussian blur to the images to evaluate how this would affect the performance of each model. 
Moreover, a Local Binary Pattern (LBP) was employed to extract features from the images for the KNN and Decision Tree models. Next, for the SVM a 
Histogram of Oriented Gradients (HOG) was implemented to extract a gradient map from the images.


## Improvements
- [ ] Show models architecture.
- [ ] Translate reports to English.
- [ ] Organize each model in a different file.

