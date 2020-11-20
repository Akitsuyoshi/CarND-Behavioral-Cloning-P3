# Import modules
import csv
import random
from numpy.core.fromnumeric import shape
from skimage.util import random_noise
from skimage.transform import rotate
from skimage.exposure import adjust_gamma
from math import ceil
import numpy as np
import tensorflow as tf
from scipy import ndimage
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Dropout, Cropping2D


csv_file = '../../../opt/carnd_p3/data/driving_log.csv'
img_folder = '../../../opt/carnd_p3/data/IMG/'
samples = []
with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    samples = [line for line in reader]

# Get train and validation samples
train_samples, validation_samples = train_test_split(samples[1:], test_size=0.25)


def get_img(source_path):
    current_path = img_folder + source_path.split('/')[-1]
    return ndimage.imread(current_path)


def get_images_and_labels(samples, is_augment=False):
    images = []
    steering_measurements = []

    for line in samples:
        # Images are Center, Left, Right in this order
        images_crl = [get_img(line[0]), get_img(line[1]), get_img(line[2])]

        # Steering
        correction = 0.2  # Hyperparam for tuning left and right image steering
        steering_center = float(line[3])
        steering_left = steering_center + correction
        steering_right = steering_center - correction
        steering_clr = [steering_center, steering_left, steering_right]

        # Images are Center, Left, Right in this order
        images.extend(images_crl)
        steering_measurements.extend(steering_clr)

        if not is_augment:
            continue

        # Horizontal flipped
        flipped_images = map(np.fliplr, images_crl)
        flipped_steering = map(lambda x: -x, steering_clr)
        images.extend(list(flipped_images))
        steering_measurements.extend(list(flipped_steering))

        if steering_center == 0.0:
            continue

        # Random noise
        noised_images = map(random_noise, images_crl)
        images.extend(list(noised_images))
        steering_measurements.extend(steering_clr)

        # Random inversion
        inversioned_images = map(np.invert, images_crl)
        images.extend(list(inversioned_images))
        steering_measurements.extend(steering_clr)

        # Random rotation
        rotated_images = map(lambda img: rotate(img, 5), images_crl)
        images.extend(list(rotated_images))
        steering_measurements.extend(steering_clr)

        # Blurred
        blurred_images = map(lambda img: ndimage.gaussian_filter(img, 3), images_crl)
        images.extend(list(blurred_images))
        steering_measurements.extend(steering_clr)

        # Random briteness
        britened_images = map(lambda img: adjust_gamma(img, gamma=random.uniform(0, 2), gain=1.), images_crl)
        images.extend(list(britened_images))
        steering_measurements.extend(steering_clr)

    return images, steering_measurements


def generator(samples, batch_size=64, is_augment=False):
    n_samples = len(samples)
    while 1:  # Loop forever so that generator never terminates
        shuffle(samples)
        for offset in range(0, n_samples, batch_size):
            batch_samples = samples[offset: offset+batch_size]
            images, steering = get_images_and_labels(batch_samples, is_augment)

            X_train = np.array(images)
            y_train = np.array(steering)

            yield shuffle(X_train, y_train)


# Hyperparam
batch_size = 64
input_size = 139

# Coroutine for train and validation samples
train_generator = generator(train_samples, batch_size=batch_size, is_augment=True)
validation_generator = generator(validation_samples, batch_size=batch_size)

# Using transfer leaning by resnet50
model = ResNet50(weights='imagenet', include_top=False,
                 input_shape=(input_size, input_size, 3))
model.pop()

for layer in model.layers:
    layer.trainable = False

road_input = input(shape=(160, 320, 3))
resized_input = Lambda(lambda img: tf.image.resize_images(img, (input_size, input_size)))(road_input)

# Here is the definition of a model
model = Sequential([
    Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)),
    Cropping2D(cropping=((60, 20), (0, 0))),
    Conv2D(3, (2, 2), activation='relu',
           kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1)),
    MaxPooling2D((2, 2)),
    Conv2D(16, (2, 2), activation='relu',
           kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (2, 2), activation='relu',
           kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (2, 2), activation='relu',
           kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(50, activation='relu', kernel_regularizer='l2'),
    Dense(10, activation='relu', kernel_regularizer='l2'),
])

model.summary()
# Dropout(0.3),
# Dense(100, activation='relu', kernel_regularizer='l2'),
# Dense(1)  # outputs a single continuous numeric value

# # Set optimizer and loss function
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# # Train model
# model.fit_generator(train_generator,
#                     steps_per_epoch=ceil(len(train_samples)/batch_size),
#                     validation_data=validation_generator,
#                     validation_steps=ceil(len(validation_samples)/batch_size),
#                     epochs=8,
#                     verbose=1)

# # Save model
# model.save('model.h5')
