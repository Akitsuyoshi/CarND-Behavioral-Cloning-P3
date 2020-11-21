import random
from skimage.util import random_noise
from skimage.transform import rotate
from skimage.exposure import adjust_gamma
import numpy as np
from scipy import ndimage
from sklearn.utils import shuffle


def get_img(source_path):
    img_folder = '../../../opt/carnd_p3/data/IMG/'
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

        images.extend(images_crl)
        steering_measurements.extend(steering_clr)

        # Augmentation happens only when training
        if not is_augment:
            continue

        # Horizontal flipped
        images.extend(list(map(np.fliplr, images_crl)))
        steering_measurements.extend(list(map(lambda x: -x, steering_clr)))

        # Most frequent class image doesn't need much augmentation
        if steering_center == 0.0:
            continue

        # Random noise
        images.extend(list(map(random_noise, images_crl)))
        steering_measurements.extend(steering_clr)

        # Random rotation
        images.extend(list(map(lambda img: rotate(img, 15), images_crl)))
        steering_measurements.extend(steering_clr)

        # Random briteness
        images.extend(list(map(lambda img: adjust_gamma(img, gamma=random.uniform(0, 3), gain=1.), images_crl)))
        steering_measurements.extend(steering_clr)

    return images, steering_measurements


def generator(samples, batch_size=64, is_augment=False):
    while True:  # Loop forever so that generator never terminates
        shuffle(samples)

        for offset in range(0, len(samples), batch_size):
            batch_samples = samples[offset: offset+batch_size]
            images, steering = get_images_and_labels(batch_samples, is_augment)

            yield shuffle(np.array(images), np.array(steering))
