## Behavioral Cloning Project

This is my fourth project of [Self-Driving Car Engineer nanodegree program](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013) in udacity.

You can see my [first](https://github.com/Akitsuyoshi/CarND-LaneLines-P1), [second](https://github.com/Akitsuyoshi/CarND-Advanced-Lane-Lines), and [third](https://github.com/Akitsuyoshi/CarND-Traffic-Sign-Classifier-Project) projects as well.

## Table of Contents

- [Behavioral Cloning Project](#behavioral-cloning-project)
- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Pipeline](#pipeline)
  - [0 Dependencies](#0-dependencies)
  - [1 Run the trained car in the simulation](#1-run-the-trained-car-in-the-simulation)
  - [2 Prepare the training / validation sets](#2-prepare-the-training--validation-sets)
  - [3 Design a model](#3-design-a-model)
    - [VGG16 setting](#vgg16-setting)
    - [Model architecture](#model-architecture)
  - [4 Training strategy](#4-training-strategy)
  - [5 Output](#5-output)
  - [6 Summary](#6-summary)
- [Discussion](#discussion)
  - [Problems during my implementation](#problems-during-my-implementation)
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

This repo includes the following files.

| File     | Description |
|:--------:|:-----------:|
| [model.py](./model.py) | the script to create and train the model|
| [driving.py](./drive.py)| the script to drive the simulating car in autonomous mode|
| [generator.py](./generator.py)| the script to prepare datasets while training a model|
| [model.h5](./model.h5) | a trained convolution neural network|
| [run1.mp4](./run1.mp4)| a output video from testing model in the siumulation|
|README.md| this file, a summary of the project|

***Note: This repo doesn't contain datasets.***

After feeding good datasets to the well-defined model, the model trains its input and then simulate a driver behavior well. An output video, that simulation from a car perspective, looks like:

![alt text][image0]

[//]: # (Image References)

[image0]: ./examples/run1.gif "Expected output"
[image1]: ./examples/hist1.png "Histogram"
[image2]: ./examples/hist2.png "Histogram2"
[image3]: ./examples/vgg16.png "VGG16"
[image4]: ./examples/vgg16_mark.png "VGG16 Marked"
[image5]: ./examples/hist3.png "Histogram3"
[image6]: ./examples/augmentation.png "Augmentation"
[image7]: ./examples/angles.png "3Angles"

---

## Pipeline

### 0 Dependencies

This project requires:

- [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)
- [Udacity self-driving car simulator](https://github.com/udacity/self-driving-car-sim)

I used the starter kit with docker to set it up. If you use mac and docker was installed successfully, you can run jupyter notebook on your local machine by this command below.

```sh
docker run -it --rm --entrypoint "/run.sh" -p 8888:8888 -v `pwd`:/src udacity/carnd-term1-starter-kit
```

***Note: I got an error while making a model in keras. This issue is related to tensorflow-gpu v1.3 problem. The detail for that can be found at [this link](https://stackoverflow.com/questions/49081129/keras-multi-gpu-model-error-swig-python-detected-a-memory-leak-of-type-int64).
To deal with that, I updated my env by this script.***

```sh
pip install --upgrade tensorflow-gpu==1.4.1
```

### 1 Run the trained car in the simulation

*If you don't wanna test the model, you can skip this step.*

Using the Udacity simulator and my [drive.py](./drive.py) file, the car can be driven autonomously around the track by executing

```sh
python drive.py model.h5
```

### 2 Prepare the training / validation sets

I first look at the deviation of datasets by `matplotlib.pyplot.hist`. The hist looked like:

![alt text][image1]

It shows that 0 angle images exist too much compared to other angle images. So I filtered to remove 80% of 0 angle image, and then the histogram changed to:

![alt text][image2]

Seems the filtering improved a bit deviation of datasets.

And I use `sklearn.model_selection.train_test_split` to get training and validating sets. That split datasets 80% for training, and 20% for validation sets.

### 3 Design a model

#### VGG16 setting

I made use of [VGG16](https://keras.io/api/applications/vgg/#vgg16-function) in keras for transfer learning. VGG16 serves as feature extraction in this project. You can see the detail of that model at [this link](https://arxiv.org/pdf/1409.1556.pdf). I use VGG16 as a pretrained model with imagenet because it works well even on small datasets like this project. In addition to that, I chose transfer learning to reduce training time and expect high accuracy using state of the art technology. Sounds cool, isn't it?

The below picture describes VGG16 architecture:

![alt text][image3]

To set it up for the later transfer learning, I removed the last three fully connected layers in VGG16. Generally, the first layers in a CNN model detected edges or abstract features of datasets, on the other hand, later layers detect its input specific features. So I made the last two convolutional layers trainable whereas the first three layers are not trainable so the VGG16 model can learn this project datasets well.

Those setting images are like:

![alt text][image4]

The above VGG16 pictures show its original input size as `(224, 224, 64)`. However, I set it to `(96, 96, 3)` because it reduces training time more than the original one.

#### Model architecture

My final model consisted of the following layers:

| Layer             |     Description                   |
|:---------------------:|:---------------------------------------------:|
| Input             | 160x320x3 RGB image                 |
| Crop        | Crop input, outputs 75x320x3    |
| Rambda(Resize)      | Resize input, outputs 96x96x3                   |
| Rambda(Preprocess)      | VGG16-specific preprocess input         |
| VGG16       |      |
| Average pooling         | 2x2 stride,  outputs 7x7x512          |
| Fully connected   | outputs 512                   |
| RELU            |                             |
| Dropout             | 0.8 remains|
| Fully connected   | outputs 512                 |
| RELU            |                             |
| Dropout             | 0.8 remains|
| Fully connected   | outputs 1                     |

I chose `Adam` as an optimizer, and `mean squared error` as an error function.

In the layers before VGG16, the model crop input at the first process so that it learns only interesting part in input data.

A bit confusing part is a preprocessing layer. I apply the same input preprocessing method as that of VGG16, which is `keras.applications.vgg16.preprocess_input`. Normalization of input data is done by this preprocessing layer in the model.

In the layers after VGG16, three fully connected layers are followed. Dropout layers exist between each fully connected layer. Also, I chose L2 regularization to the third and second last layers for preventing overfitting.

### 4 Training strategy

The final Hyperparameters for training are:

- 0.005 learning **rate**
- 32 batch size
- 5 Epocs

I've done fine tuning those parameters every after testing the trained model in the simulator. I realized that no matter how less training loss was, it doesn't mean that the model learned and worked well in the test, simulation environment. So I always set training epochs to 1 to make sure that the current change is on the right track when I train the model.

During the training, I first faced underfitting results because of less trainable parameters in the model. So I added additional layers after the VGG16 layer, and make the last two convolutional layers in VGG16. It eliminated the underfitting, but once I increase the epoch number and datasets, that caused overfitting. To deal with that, I put dropout layers and put L2 regularization.

I also normalized steering(label) value, by `round(float(steering) * 50) / 50` to make all steering in 0.2 increments.

At this point, however, the model still couldn't curve the first left turn in track 1. So I see the datasets deviation, and filtered 80% of 0 angle images. And then I do image augmentation to increase less label(angle) images to make model train curve images more.

These are image augmentations that I implemented in [generator.py](./generator.py).

- Noised, using `skimage.util.random_noise`
- Rotation, using `skimage.transform.rotate`
- Blurred, using `scipy.ndimage.gaussian_filter`
- Random Briteness, using `skimage.exposure.adjust_gamma`
- Horizontal flip, using `np.fliplr`

The above-augmented images look like the pic below. The images are original, noised, rotated, blurred, random brightness, and horizontal flip in an order from the upper left to bottom right.
![alt text][image6]

On top of those above augmentations, I use three camera angle images in datasets. These images can be a good input for the model to learn recovering from the left side and right sides of the road back to the center.

Three camera angle images are:
![alt text][image7]

The final total number of datasets increase from `4572` to `42973`.

Before datasetse histogram:
![alt text][image2]

After image augmentation and adding three angle images:
![alt text][image5]

I prepared a good enough amount of datasets for the training.

To improve the training process, I use `keras.callbacks. ModelCheckpoint` to save only good results after each epoch.

### 5 Output

The output video can be found at [run1.mp4](./run1.mp4).

### 6 Summary

Here are the steps that I took.

- Prepare the datasets
- Filter too much label image(0 angle image in this project)
- Split datasets into training and validating sets
- Make a model, using VGG16 for transfer learning
- Train the model
- Fine tune the model
- Test the model in the simulator
- Got a bad result
- Deal with underfitting / overfitting
- Add augmented and three angle images
- Train again

I iterate steps from training to testing until getting to the final model and expected outputs, which model can predict good steering value and the car in autonomous mode drive 1 lap around track 1.

## Discussion

### Problems during my implementation

My first misunderstanding was unfreezing all layers in VGG16. I thought that was an appropriate way because I didn't know well transfer learning at that point. However, I noticed that it didn't make the model learn at all. So I searched [article](https://keras.io/guides/transfer_learning/#build-a-model) in keras to understand how transfer learning works.

Another miss was to train all datasets without any augmentation or preprocessing. That was a waste of time on second thoughts.

The last miss was a strategy when training the model. I always set epoch 5 but I should have set it epoch 1 not to get lost my implementation. The thing is feedback of the model is important, but if I keep training without a test, driving in the simulation in this case, I can easily get lost. To prevent that lost time, I decided to set epoch 1 at the beginning of my implementation.

### Improvements to pipeline

- Change the way to preprocess datasets
- Try another image augmentations

### Future Feature

- Test track 2 to see how well my current model predicts.
- Apply another model instead of VGG16
- Prepare the datasets for training when implementing, which should be much smaller than the original datasets. That's good for quick feedback when designing a model.
  
---

## References

- [Transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/#build-a-model)
- [Paper about VGG16](https://arxiv.org/pdf/1409.1556.pdf)
- [memory leak error in keras](https://stackoverflow.com/questions/49081129/keras-multi-gpu-model-error-swig-python-detected-a-memory-leak-of-type-int64)
  
---

## Author

- [Tsuyoshi Akiyama](https://github.com/Akitsuyoshi)
