# Import modules
import csv
import random
from skimage.util import random_noise
from skimage.transform import rotate
from skimage.exposure import adjust_gamma
from math import ceil
import numpy as np
import tensorflow as tf
from scipy import ndimage
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense, Lambda, Dropout, Input, GlobalAveragePooling2D


csv_file = '../../../opt/carnd_p3/data/driving_log.csv'
img_folder = '../../../opt/carnd_p3/data/IMG/'
samples = []
with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    samples = [line for line in reader]

# Get train and validation samples
train_samples, validation_samples = train_test_split(samples[1:], test_size=0.2)


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

        if not is_augment or steering_center == 0.0:
            continue

        # Horizontal flipped
        images.extend(list(map(np.fliplr, images_crl)))
        steering_measurements.extend(list(map(lambda x: -x, steering_clr)))

        # Random noise
        images.extend(list(map(random_noise, images_crl)))
        steering_measurements.extend(steering_clr)

        # Random rotation
        images.extend(list(map(lambda img: rotate(img, 10), images_crl)))
        steering_measurements.extend(steering_clr)

        # Blurred
        images.extend(list(map(lambda img: ndimage.gaussian_filter(img, 3), images_crl)))
        steering_measurements.extend(steering_clr)

        # Random briteness
        images.extend(list(map(lambda img: adjust_gamma(img, gamma=random.uniform(0, 2), gain=1.), images_crl)))
        steering_measurements.extend(steering_clr)

    return images, steering_measurements


def generator(samples, batch_size=32, is_augment=False):
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
batch_size = 32

# Coroutine for train and validation samples
train_generator = generator(train_samples, batch_size=batch_size, is_augment=True)
validation_generator = generator(validation_samples, batch_size=batch_size, is_augment=False)


# ------------------------------------------------
# Following is the definition of a model
train_samples_shape = (160, 320, 3)
model_input_shape = (56, 112, 3)  # original shape is 224 * 224

# Using resnet50 for transfer learning
resnet = ResNet50(weights='imagenet', include_top=False,
                  input_shape=model_input_shape)
resnet.layers.pop()

for layer in resnet.layers:
    layer.trainable = False

# model input placeholder
model_inputs = Input(shape=train_samples_shape)
normalized_inputs = Lambda(lambda img: (img / 255.0) - 0.5)(model_inputs)
cropped_inputs = Lambda(lambda img: tf.image.crop_to_bounding_box(img, 60, 0, 80, train_samples_shape[2]))(normalized_inputs)
resized_inputs = Lambda(lambda img: tf.image.resize_images(img, (model_input_shape[0], model_input_shape[1])))(cropped_inputs)
resnet_outputs = resnet(resized_inputs)

# For model outputs
model_outputs = GlobalAveragePooling2D()(resnet_outputs)
model_outputs = Dropout(0.2)(model_outputs)
model_outputs = Dense(50, activation='relu', kernel_regularizer='l2')(model_outputs)
model_outputs = Dense(10, activation='relu', kernel_regularizer='l2')(model_outputs)
model_outputs = Dense(1)(model_outputs)  # Outputs a single value
model = Model(inputs=model_inputs, outputs=model_outputs)
model.summary()


# Set optimizer and loss function
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# Train model
model.fit_generator(train_generator,
                    steps_per_epoch=ceil(len(train_samples)/batch_size),
                    validation_data=validation_generator,
                    validation_steps=ceil(len(validation_samples)/batch_size),
                    epochs=10,
                    verbose=1)

# Save model
model.save('model.h5')
print('Model was saved successfully')
