# Import modules
import csv
from generator import generator
import tensorflow as tf
from math import ceil
from sklearn.model_selection import train_test_split
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense, Lambda, Dropout, Input, GlobalAveragePooling2D
from keras.optimizers import Adam


def get_model():
    """Returns model architecture, get detail by `returned_model.summary()`"""
    # Hyperparams
    train_samples_shape = (160, 320, 3)
    model_input_shape = (56, 56, 3)  # original shape is 224 * 224

    # ResNet50 setting
    resnet = ResNet50(weights='imagenet', include_top=False,
                      input_shape=model_input_shape)
    resnet.layers.pop()
    resnet.trainable = False

    # model input placeholder
    model_inputs_placeholder = Input(shape=train_samples_shape)

    # For model inputs
    model_inputs = Lambda(lambda img: (img / 255.0) - 0.5)(model_inputs_placeholder)
    model_inputs = Lambda(lambda img: tf.image.crop_to_bounding_box(img, 50, 0, 90, train_samples_shape[1]))(model_inputs)
    model_inputs = Lambda(lambda img: tf.image.resize_images(img, (model_input_shape[0], model_input_shape[1])))(model_inputs)

    # For model outputs
    model_outputs = resnet(model_inputs)
    model_outputs = GlobalAveragePooling2D()(model_outputs)
    model_outputs = Dense(64, activation='relu', kernel_regularizer='l2')(model_outputs)
    model_outputs = Dropout(0.2)(model_outputs)
    model_outputs = Dense(10, activation='relu', kernel_regularizer='l2')(model_outputs)
    model_outputs = Dropout(0.2)(model_outputs)
    model_outputs = Dense(1)(model_outputs)  # Outputs a single value

    return Model(inputs=model_inputs_placeholder, outputs=model_outputs)


if __name__ == "__main__":
    csv_file = '../../../opt/carnd_p3/data/driving_log.csv'
    samples = []

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        samples = [line for line in reader]

    # Get train and validation samples
    train_samples, validation_samples = train_test_split(samples[1:], test_size=0.2)

    # Hyperparam
    batch_size = 64

    # Coroutine(lazy loading) for train and validation samples
    train_generator = generator(train_samples, batch_size=batch_size, is_augment=True)
    validation_generator = generator(validation_samples, batch_size=batch_size, is_augment=False)

    # Get model architecture
    model = get_model()
    model.summary()

    # Set optimizer and loss function
    model.compile(loss='mse', optimizer=Adam(3e-3))

    # Train model
    model.fit_generator(train_generator,
                        steps_per_epoch=ceil(len(train_samples)/batch_size),
                        validation_data=validation_generator,
                        validation_steps=ceil(len(validation_samples)/batch_size),
                        epochs=5,
                        verbose=1)

    # Save model
    model.save('model.h5')
