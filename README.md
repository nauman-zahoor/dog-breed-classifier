# Dog Breed Classifier


## Project Overview
This project utilizes CNNs and consepts of transfer learning to first train multiple classifiers to identify Human and dogs in images and then to classify dog breeds in images. Transfer learning is a technique where we use pre trained model's initial layers as-is and fine tune the last layers for final goal. Here we make use of Resnet-50, InceptionV3, VGG16 and other pre trained models. This gives us much beeter results if we were to train a  classifier from scratch as we dont have to train for a very long time or on a very powerful machine. In seocnd part of the project, a webapp is built that makes use of trained models. Webapp asks for an image and checks if a human or dog is present in image. If Dog is present, it identifies Dog breed. If Human is present, it identifies what dog breed does the human resembles. If human or dog are not found in image, it returns a message stating as such. 

## File descriptions
This project contains a jupyter notebook named **dog_app.ipynb** that has all the data analysis and machine learning pipeline.
Webapp for project is in **Webapp** folder.

## Installation
Install required libraries using requirements file using the following instruction.
```
pip install -r requirements.txt 
```
## Instructions to run webapp
Step 1. go to Webapp's directory
```
cd Webapp
```
Step 2. Run following to start webapp
```
python run.py
```
By Default the app runs at port 3001 and can be accessed at http://0.0.0.0:3001/

## Problem Statement
### Topic: Classification of input images to identify dog breeds. 
### Description: 
Goal of this project is to identify and classifiy dog breeds from input image. Before we can classify dog breeds, we also need to check if there is even a dog present in the image or not so we need to have another classifier that does this task. We also check if human is present in the image so that we can identify resembling dog breed. After confirming presence of a human or a dog, we need to train a second classfier that classifies image in to respective 133 dog breeds. 
## Dataset
Dog images to train model can be found at following  [link](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). Dataset is composed of dog images belonging tp 133 classes. Data has already been split into Train, Test and Validation sets. Train and Validation sets are used during model training final model performance is evaluated on Test set. 
Below is an overview of dataset.
* There are 133 total dog categories.
* There are 8351 total dog images.
* There are 6680 training dog images.
* There are 835 validation dog images.
* There are 836 test dog images.

### Metrics and Model Evaluation
**Accuracy** on testset is used as evaluation metric. Test set is used for final model evaluation.

## Proposed Solution 
* Step 1: Detect Humans
* Step 2: Detect Dogs
* Step 3: Compare preformance of different CNNs architectures to Classify Dog Breeds 
    * a: CNN From Scratch
    * b: Transfer learning using VGG16
    * c: Transfer lerning using InceptionV3
* Step 4: Dog Breed Detection Algorithm
* Findings

### Step 1: Detect Humans
OpenCV's implementation of [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) is used to detect human faces in images. OpenCV provides many pre-trained face detectors, stored as XML files on github. Haarcascades directory contains a pre-trained face detector. 
This detector is able to identify humans in images at an acfuracy of 100% but it suffers when a dog image is provided as it classifies 11% of dog images as humans. 
So this opencv classifier is used only to detect humans and for dog detection, another classifier is used.
### Step 2: Detect Dogs
 A pre-trained [ResNet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006) model is used to detect dogs in images.
 Resnet is trained on [ImageNet](http://www.image-net.org/) that contains over 10 million URLs, each linking to an image containing an object from one of 1000 categories. Given an image, this pre-trained ResNet-50 model returns a prediction (derived from the available categories in ImageNet) for the object that is contained in the image.
 In Resnet, the categories corresponding to dogs appear in an uninterrupted sequence and correspond to dictionary keys from 151 to 268, inclusive, to include all categories from 'Chihuahua' to 'Mexican hairless'. Thus, in order to check to see if an image is predicted to contain a dog by the pre-trained ResNet-50 model, we need only check if the predicted labels are a value between 151 and 268.
 This model was able to detect dogs in images with an accuracy of 100%.
 
### Step 3: Compare preformance of different CNNs architectures to Classify Dog Breeds 
Now that we have identified that if a dog or human is present in the image or not, next step is to identify its breed. CNNs are used for this purpose too. We trained a CNN from scratch and compared its performance with pretrained CNNs, VGG16 and InceptionV3, to identify best performing one.
#### a: CNN From Scratch
A CNN model is build from scratch with the following architecture.
![CNN arch](https://github.com/nauman-zahoor/dog-breed-classifier/blob/main/Webapp/screenshots/cnn_from_scratch.png?raw=true)
Model was trained for 10 epochs and yielded and accuracy of 4.0670%.

#### b: Transfer learning using VGG16
To reduce training time without sacrificing accuracy, transfer learning is used. VGG16 is used as pretrained model to obtain bottleneck features of each input image. A new model is created and trained consisting of GlobalAveragePooling layer and final Dense layer to predict 133 dog breeds.
Model achieved accuracy of 53.2297% on test set.

#### c: Transfer lerning using InceptionV3
InceptionV3 is also used as pretrained model to calculate bottleneck featues which is then fed to a second model consisting of GlobalAveragePooling layer and final Dense layer to predict 133 dog breeds.
This model achieved accuracy of 82.1770% on test set which is highest than all the rest.
#### Step 4: Dog Breed Detection Algorithm
The final algorithm that is used to predict dog breeds works in following way.
1. First takes an input image.
2. It then tries to detect if a human is present in image or not using the Human face detector that was built in step-1. If Human is found then algo uses InceptionV3 model to find the resembling dog breed to input human. 
3. If human was not found earlier then algo uses Resenet model used in step-2 to detect if any dog is present in image. If Dog is found then algo uses InceptionV3 model to find the dog breed. 
4. If Human and Dog were not detected then algo return a string stating such.

#### Conclusion/ Findings
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

