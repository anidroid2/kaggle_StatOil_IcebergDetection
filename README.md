# StatOil Iceberg Detection Challenge (Kaggle Competition)

Competition Link: https://www.kaggle.com/c/statoil-iceberg-classifier-challenge

This repository contains base code for that can be advanced further to achieve good results in the competition. 

Keras model of Convolution Neural Network to calssify images into Iceberg or Ship. Training and Test set images is provided by the competition organizers and can be downloaded from the competition website. 

### File Description: 

data1.py : Load training data in python environment

data2.py : Load test data in python environment

main.py  : Implementation of the Keras CNN model and generation of output submission file at the end

EDA.ipynb: Exploratory Data Analysis performed in iPythonNotebook.  

### Accuracy and Loss Graphs

The program generates the graphs by storing Accuracy and Loss at every step.


![](1.png?raw=true)

The frequent fluctations in the validation set is due to the reason that we use minibatch learning. After 250 iterations, we see that it is overfitting the train data.

![](2.png?raw=true)

With some tuning it is possible to reach 85% accuracy in validation set. 


![](3.png?raw=true)

With some more tweaks, validation set accuracy is almost reaching 90%.