# Dog Breed Classifier


## Project Overview and Motivation
This project utilizes CNNs and consepts of transfer learning to first train multiple classifiers to identify Human and dogs in images and then to classify dog breeds in images. Transfer learning is a technique where we use pre trained model's initial layers as-is and fine tune the last layers for final goal. Here we make use of Resnet, Inception, VGG and other pre trained models. This gives us much beeter results if we were to train a  classifier from scratch as we dont have to train for a very long time or on a very powerful machine. In seocnd part of the project, a webapp is built that makes use of trained models. Webapp asks for an image and checks if a humman or dog is present in image. If Dog is present, it identifies Dog breed. If Human is present, it identifies what dog breed does the human resembles. If human or dog are not found in image, it returns a message stating as such.

## File descriptions
This project contains a jupyter notebook named **dog_app.ipynb** that has all the data analysis and machine learning pipeline.
Webapp for project is in **Webapp** folder.

## Problem Statement
Goal of this project is to identify and classifiy dog breeds from input image. Before we can classify dog breeds, we also need to check if there is even a dog present in the image or not.
This problem would require a model to take input as an image. CNNs are best known for their ability to automatically extract featres from images with out specifically defining. 
Data set is divided in three groups Train, Validation and Test. Train and Validation sets are used during model training final model performance is evaluated on Test set. 
Accuracy on test set is used as final evaluation metric.

## Installation
Install required libraries using requirements file using the following instruction.
```
pip install -r requirements.txt 
```
## Instructions to run webapp

go to Webapp's directory
```
cd Webapp
```

Run following to start webapp
```
python run.py
```
By Default the app runs at port 3001 and can be accessed at http://0.0.0.0:3001/

## Findings
* Resent model is used for identifying if dog is present in image or not.
* OpenCV's implementation of Haar feature-based cascade classifiers is used to detect if human is present in image or not.
* Final Inception Classifier used for dog breed classification reached test accuracy of 82.1770% which is much higher than classifier trained form scratch.
* Model had a hard time and would often mismatch and misclassify dogbreeds that share a lot of same visual features i.e. hair texture, hair color etc. This can be improved by adding more examples of such classes and retraining.
* Model can be improved by providing augmented data to cater for noisy images.

## App Screenshots
App main page:

![Main page](https://github.com/nauman-zahoor/dog-breed-classifier/blob/main/Webapp/screenshots/webapp_screenshot1.png?raw=true)


App Historical Predictions:
![Historical Predictions](https://github.com/nauman-zahoor/dog-breed-classifier/blob/main/Webapp/screenshots/webapp_screenshot2.png?raw=true)

## Authors

* [Nauman Zahoor](https://github.com/nauman-zahoor/)

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Acknowledgements

* [Udacity](https://www.udacity.com/)

