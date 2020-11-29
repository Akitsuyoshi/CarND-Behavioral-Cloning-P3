# Import modules
import csv
import random
from generator import generator
import tensorflow as tf
from math import ceil
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.layers import Dense, Lambda, Dropout, Input, GlobalAveragePooling2D, Cropping2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


def get_model():
    """Returns model architecture, get detail by `returned_model.summary()`"""
    # Hyperparams
    train_samples_shape = (160, 320, 3)
    model_input_shape = (96, 96, 3)

    # VGG16 setting
    vgg16 = VGG16(weights='imagenet', include_top=False,
                  input_shape=model_input_shape)
    vgg16.trainable = True
    for layer in vgg16.layers:
        if layer.name in ['block5_conv1', 'block4_conv1']:
            layer.trainable = True
        else:
            layer.trainable = False
    vgg16.summary()

    # model input placeholder
    model_inputs_placeholder = Input(shape=train_samples_shape)

    # For model inputs
    model_inputs = Cropping2D(cropping=((60, 25), (0, 0)))(model_inputs_placeholder)
    model_inputs = Lambda(lambda img: tf.image.resize_images(img, (model_input_shape[0], model_input_shape[1])))(model_inputs)
    model_inputs = Lambda(lambda img: preprocess_input(img))(model_inputs)

    # For model outputs
    model_outputs = vgg16(model_inputs)
    model_outputs = GlobalAveragePooling2D()(model_outputs)
    model_outputs = Dense(512, activation='relu', kernel_regularizer='l2')(model_outputs)
    model_outputs = Dropout(0.2)(model_outputs)
    model_outputs = Dense(512, activation='relu', kernel_regularizer='l2')(model_outputs)
    model_outputs = Dropout(0.2)(model_outputs)
    model_outputs = Dense(1)(model_outputs)  # Outputs a single value

    return Model(inputs=model_inputs_placeholder, outputs=model_outputs)


if __name__ == "__main__":
    csv_file = '../../../opt/carnd_p3/data/driving_log.csv'
    model_file_path = 'model.h5'
    samples = []

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        samples = [line for line in reader]
        samples = samples[1:]

    filtered_samples = list(filter(lambda line: float(line[3]) != 0.0 or random.uniform(0, 1) > 0.8, samples))

    # Get train and validation samples
    train_samples, validation_samples = train_test_split(filtered_samples, test_size=0.2)

    # Hyperparam
    batch_size = 32

    # Coroutine(lazy loading) for train and validation samples
    train_generator = generator(train_samples, batch_size=batch_size, is_augment=True)
    validation_generator = generator(validation_samples, batch_size=batch_size, is_augment=False)

    # Get model architecture
    model = get_model()
    model.summary()

    # Set optimizer and loss function
    model.compile(loss='mse', optimizer=Adam(5e-3))

    # Callback for saving best result model
    model_checkpoint = ModelCheckpoint(filepath=model_file_path,
                                       monitor='val_loss',
                                       save_best_only=True,
                                       save_weights_only=True,
                                       mode='min',
                                       period=1)

    # Train model
    model.fit_generator(train_generator,
                        steps_per_epoch=ceil(len(train_samples)/batch_size),
                        validation_data=validation_generator,
                        validation_steps=ceil(len(validation_samples)/batch_size),
                        epochs=5,
                        verbose=1,
                        callbacks=[model_checkpoint])
