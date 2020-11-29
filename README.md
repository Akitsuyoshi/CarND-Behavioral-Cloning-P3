## Behavioral Cloning Project

This is my fourth project of [Self-Driving Car Engineer nanodegree program](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013) in udacity.

You can see my [first](https://github.com/Akitsuyoshi/CarND-LaneLines-P1), [second](https://github.com/Akitsuyoshi/CarND-Advanced-Lane-Lines), and [third](https://github.com/Akitsuyoshi/CarND-Traffic-Sign-Classifier-Project) projects as well.

## Table of Contents

- [Behavioral Cloning Project](#behavioral-cloning-project)
- [Table of Contents](#table-of-contents)
- [Overview](#overview)
  - [0 Dependencies](#0-dependencies)
  - [1 How to run the trained car in the simulation](#1-how-to-run-the-trained-car-in-the-simulation)
  - [2 Prepare the training / validation sets](#2-prepare-the-training--validation-sets)
  - [3 Design a model](#3-design-a-model)
    - [VGG16 setting](#vgg16-setting)
    - [Model architecture](#model-architecture)
  - [4 Training strategy](#4-training-strategy)
  - [5 Output](#5-output)
  - [6 Summary](#6-summary)
- [Discussion](#discussion)
  - [Problem during my implementation](#problem-during-my-implementation)
  - [Improvements to pipeline](#improvements-to-pipeline)
  - [Future Feature](#future-feature)
- [References](#references)
- [Author](#author)

---

## Overview

The goals / steps of this project are the following:
- Use the Udacity [sample driving dataset](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)
- Build, a convolution neural network in Keras and Tensorflow that predicts steering angles from images
- Train and validate the model with a training and validation set
- Test that the model successfully drives around track one without leaving the road
- Summarize the results

This repo includes following files.

| File     | Description |
|:--------:|:-----------:|
| model.py | the script to create and train the model|
| driving.py| the script to drive the simulating car in autonomous mode|
| model.h5 | a trained convolution neural network|
| run1.mp4| a output video from testing model driving accuracy in the simulation |
|README.md| a summary of the project|

***Note: This repo doesn't contain datasets.***

[//]: # (Image References)

[image1]: ./examples/hist1.png "Histogram"
[image2]: ./examples/hist2.png "Histogram2"
[image3]: ./examples/vgg16.png "VGG16"
[image4]: ./examples/vgg16_mark.png "VGG16 Marked"

---

### 0 Dependencies

This project requires:

- [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)
- [Udacity self-driving car simulator](https://github.com/udacity/self-driving-car-sim)

I used the started kit with docker to set it up. If you use mac and docker was installed successfully, you can run jupyter notebook on your local machine by this command below.

```sh
docker run -it --rm --entrypoint "/run.sh" -p 8888:8888 -v `pwd`:/src udacity/carnd-term1-starter-kit
```

***Note: I got an error while making model in keras. This issue is related to tensorflow-gpu v1.3 problem. The detail for that can be found at [this link](https://stackoverflow.com/questions/49081129/keras-multi-gpu-model-error-swig-python-detected-a-memory-leak-of-type-int64).
To deal with that, I updated env by this script.***

```sh
pip install --upgrade tensorflow-gpu==1.4.1
```

### 1 How to run the trained car in the simulation

Using the Udacity simulator and my drive.py file, the car can be driven autonomously around the track by executing

```sh
python drive.py model.h5
```

### 2 Prepare the training / validation sets

I first look the deviation of datasets by `matplotlib.pyplot.hist`. The hist looked like:

![alt text][image1]

I filtered to remove 80% of 0 angle image, and then the histogram changed to:

![alt text][image2]

Seems it improved a bit deviation of datasets.

And I use `sklearn.model_selection.train_test_split` to get training and validating sets. That splitted datasets 80% for training, and 20% for validation sets.

### 3 Design a model

#### VGG16 setting

I made used of [VGG16](https://keras.io/api/applications/vgg/#vgg16-function) in keras for the transfer learning.

The picture below describes the VGG16 architecture:

![alt text][image3]

To set it up for the later trasfer learning, I remove last three fully connected layers in VGG16. Generally, the first layers in CNN model detected edges or abstract features of datasets, on the other hand, later layers detect problem specific features. So I made last two convolutional layers trainable whereas first three layers are not trainable so that VGG16 model can learn this project datasets well.

Those setting images are like:

![alt text][image4]

#### Model architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 160x320x3 RGB image   							|
| Crop      	| Crop input, outputs 75x320x3  	|
| Rambda			|	Resize input, outputs 96x96x3 									|
| Rambda	   	| VGG16-specific preprocess input  				|
| VGG16       |      |
| Average pooling	      	| 2x2 stride,  outputs 7x7x512   				|
| Fully connected		| outputs 512  									|
| RELU				    |           									|
| Dropout   	      	| 0.8 remains|
| Fully connected		| outputs 512									|
| RELU				    |            									|
| Dropout   	      	| 0.8 remains|
| Fully connected		| outputs 1   									|

I chose `Adam optimizer`, and `mean squared error` as error function.

A bit confusing part is preprocessing layer. I apply the same input preprocessing method as that of VGG16. Normalization of input data is done by this preprocessing layer in the model.

In the layers after VGG16, three fully connected layers are followed. Dropout layers exists between the fully layers. Also, I chose L2 regularization to the third and second last for preventing overfitting.



### 4 Training strategy

Hyperparameters for training are:

- 0.005 learning rate
- 32 batch size
- 5 Epocs

### 5 Output

### 6 Summary

## Discussion

### Problem during my implementation

### Improvements to pipeline

### Future Feature

---

## References

---

## Author

- [Tsuyoshi Akiyama](https://github.com/Akitsuyoshi)
