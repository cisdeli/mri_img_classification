# Classifying levels of Alzheimer's disease using MRI and machine learning

The objective of this project is to evaluate the perfomance of different machine learning classification algorithms paired with
image processing techniques. Additionally, we will use the trained models to predict the level of dementia
based on the MRI images.

This project was developed for a Machine Learning class at the University of São Paulo (USP) in 2021. Because of this, most of the reports are in Portuguese.
I will try to translate them to English as soon as possible. For now use this markdown file as a guide to the project.

***Authors***:
* Paulo Ricardo J. Miranda, NUSP: 10133456
* Pedro Henrique Magalhães Cisdeli, NUSP: 10289804

## Relevancy of the project

Alzheimer's disease is a neurodegenerative disease that affects millions of people worldwide. The disease is known for a really complex diagnosis process, which involves a lot of different exams and specialists.
The early detection of the disease is crucial for the patient's treatment and quality of life. 
The use of machine learning algorithms to classify the level of dementia based on MRI images can help doctors to diagnose the disease more accurately and quickly.

## Dataset Information

* The dataset can be accessed on [Kaggle](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images).
* 6400 MRI images of the brain.
* Resolution is 128x128 pixels.
* Grayscale images.


An example of the images can be seen on **Figure 1**.
<figure align="center">
    <img src="assets/mri_example.jpeg" alt="MRI Image Example" width="50%">
</figure>


*Figure 1. Example of a higher resolution brain MRI Image - [Source](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection).*

### Classes
The dataset is divided into 4 classes: 
* Non-Demented.
* Very Mild Demented.
* Mild Demented.
* Moderate Demented.
  
The classes are unbalanced, with the majority of the images being classified as Non-Demented.
To account for this factor the same distribution of classes was maintained for the training and test sets, as shown on **Figure 2**.
<figure align="center">
    <img src="assets/class_dist.png" alt="Class Distribution" width="100%">
</figure>


*Figure 2. Bar graph of the original class distribution from the dataset on the left; Bar graph showing the same class distribution being kept for the train and test set on the right.*


## Models Tested
* K-Nearest Neighbors *(KNN)*.
* Decision Tree *(DT)*.
* Support Vector Machine *(SVM)*.
* Multilayer Perceptron *(MLP)*.
* Convolutional Neural Network *(CNN)8.

## Image Processing

First, different levels of Gaussian blur (**Figure 3**) were applied to the images to evaluate how this would affect the performance of each model. Moreover, a Local Binary Pattern (LBP) (**Figure 4**) was employed to extract features from the images for the KNN and Decision Tree models. Next, for the SVM a 
Histogram of Oriented Gradients (HOG) (**Figure 5** illustrates better how this method works with an image of a dog) was implemented to extract a gradient map from the images. The only treatment for the MLP and CNN models was the Gaussian blur.
<figure align="center">
    <img src="assets/noise.png" alt="Noise Distribution" width="100%">
</figure>

*Figure 3. First images on the left shows an example of the control group; The image on the middle presents an example of the low gaussian blur group; Lastly, the image on the right shows an example of an image with the highest amount of noise on this study.*

<figure align="center">
    <img src="assets/LBP.png" alt="LBP" width="25%">
</figure>

*Figure 4. Example of an output from the Local Binary Pattern algorithm.*

<figure align="center">
    <img src="assets/HOG.png" alt="LBP" width="50%">
</figure>

*Figure 5. Example of an output from the HOG algorithm. An image of a dog was used to better illustrate how the gradient map works.*

## Grid Search and Cross Validation

Grid search and cross validation methods were employed to ensure that all of the models had the best parameters possible.

## Results

Accuracy and performance of models tested on all three conditions (Control, 0.01 Noise, 0.001 Noise) can be seen on **Table 1**.

*Table 1. Accuracy and AUC of the models tested on the three conditions.*
<table>
  <tr>
    <th rowspan="2">Model</th>
    <th colspan="2">Control Images</th>
    <th colspan="2">Images with 0.01 Noise</th>
    <th colspan="2">Images with 0.001 Noise</th>
  </tr>
  <tr>
    <th>Accuracy</th>
    <th>AUC</th>
    <th>Accuracy</th>
    <th>AUC</th>
    <th>Accuracy</th>
    <th>AUC</th>
  </tr>
  <tr>
    <td>KNN</td>
    <td>0.53</td>
    <td>0.54</td>
    <td>0.34</td>
    <td>0.50</td>
    <td>0.35</td>
    <td>0.50</td>
  </tr>
  <tr>
    <td>KNN_001</td>
    <td>0.26</td>
    <td>0.50</td>
    <td>0.51</td>
    <td>0.53</td>
    <td>0.51</td>
    <td>0.50</td>
  </tr>
  <tr>
    <td>KNN_0001</td>
    <td>0.34</td>
    <td>0.50</td>
    <td>0.33</td>
    <td>0.50</td>
    <td>0.51</td>
    <td>0.53</td>
  </tr>
  <tr>
    <td>DT</td>
    <td>0.51</td>
    <td>0.54</td>
    <td>0.34</td>
    <td>0.50</td>
    <td>0.37</td>
    <td>0.48</td>
  </tr>
  <tr>
    <td>DT_001</td>
    <td>0.34</td>
    <td>0.50</td>
    <td>0.51</td>
    <td>0.53</td>
    <td>0.34</td>
    <td>0.50</td>
  </tr>
  <tr>
    <td>DT_0001</td>
    <td>0.34</td>
    <td>0.50</td>
    <td>0.51</td>
    <td>0.50</td>
    <td>0.50</td>
    <td>0.51</td>
  </tr>
  <tr>
    <td>SVM</td>
    <td>0.70</td>
    <td>0.70</td>
    <td>0.56</td>
    <td>0.58</td>
    <td>0.65</td>
    <td>0.68</td>
  </tr>
  <tr>
    <td>SVM_001</td>
    <td>0.59</td>
    <td>0.62</td>
    <td>0.56</td>
    <td>0.60</td>
    <td>0.58</td>
    <td>0.62</td>
  </tr>
  <tr>
    <td>SVM_0001</td>
    <td>0.66</td>
    <td>0.67</td>
    <td>0.58</td>
    <td>0.60</td>
    <td>0.66</td>
    <td>0.68</td>
  </tr>
  <tr>
    <td>MLP</td>
    <td>0.71</td>
    <td>0.71</td>
    <td>0.61</td>
    <td>0.67</td>
    <td>0.70</td>
    <td>0.71</td>
  </tr>
  <tr>
    <td>MLP_001</td>
    <td>0.56</td>
    <td>0.54</td>
    <td>0.68</td>
    <td>0.67</td>
    <td>0.62</td>
    <td>0.59</td>
  </tr>
  <tr>
    <td>MLP_0001</td>
    <td>0.72</td>
    <td>0.67</td>
    <td>0.52</td>
    <td>0.64</td>
    <td>0.76</td>
    <td>0.71</td>
  </tr>
  <tr>
    <td>CNN</td>
    <td>0.96</td>
    <td>0.98</td>
    <td>0.82</td>
    <td>0.91</td>
    <td>0.95</td>
    <td>0.98</td>
  </tr>
  <tr>
    <td>CNN_001</td>
    <td>0.96</td>
    <td>0.93</td>
    <td>0.93</td>
    <td>0.81</td>
    <td>0.96</td>
    <td>0.92</td>
  </tr>
  <tr>
    <td>CNN_0001</td>
    <td>0.96</td>
    <td>0.99</td>
    <td>0.91</td>
    <td>0.98</td>
    <td>0.96</td>
    <td>0.99</td>
  </tr>
</table>

## Improvements
- [ ] Show models architecture.
- [ ] Translate reports to English.
- [ ] Organize each model in a different file.

## References 

* https://repositorio.unesp.br/bitstream/handle/11449/151042/padovese_bt_me_sjrp.pdf?sequence=3&isAllowed=y

* GAION, João Pedro de Barros Fernandes. Doença de Alzheimer: saiba mais sobre a principal causa de demência no mundo. saiba mais sobre a principal causa de demência no mundo. 2020. Disponível em: https://www.informasus.ufscar.br/doenca-de-alzheimer-saiba-mais-sobre-a-principal-causa-de-demencia-no-mundo/. Acesso em: 20 abr. 2022.

* ALZHEIMER'S ASSOCIATION. Alzheimer e demência no Brasil. 20--?. Disponível em: https://www.alz.org/br/demencia-alzheimer-brasil.asp. Acesso em: 20 abr. 2022.

* NAZARÉ, Thiago Santana de, et. al. Deep Convolutional Neural Networks and Noisy Images. Disponível em: https://sites.icmc.usp.br/moacir/papers/Nazare_CIARP2017_DNN-Noise.pdf. Acesso em: 22 abr. 2022.

* CNN Model Architecture: https://ieeexplore.ieee.org/document/9215402
