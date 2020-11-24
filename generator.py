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
        img_center = get_img(line[0])
        img_left = get_img(line[1])
        img_right = get_img(line[2])
        img_clr = [img_center, img_left, img_right]

        # Steering
        correction = random.uniform(0.2, 0.25)
        steering_center = round(float(line[3]) * 50) / 50
        steering_left = steering_center + correction
        steering_right = steering_center - correction
        steering_clr = [steering_center, steering_left, steering_right]

        images.extend(img_clr)
        steering_measurements.extend(steering_clr)

        if not is_augment:
            continue

        for img, steering in zip(img_clr, steering_clr):
            images.extend([np.fliplr(img),  # Horizontal flipped
                           random_noise(img),  # Random noise
                           rotate(img, random.uniform(-15, 15)),  # Random rotation
                           ndimage.gaussian_filter(img, random.randrange(5, 17, 2)),  # Blurred
                           adjust_gamma(img, gamma=random.uniform(0, 2), gain=1.),  # Random briteness
                           ])

            steering_measurements.extend([-steering,
                                          steering,
                                          steering,
                                          steering,
                                          steering,
                                          ])

    return images, steering_measurements


def generator(samples, batch_size=64, is_augment=False):
    while True:  # Loop forever so that generator never terminates
        shuffle(samples)

        for offset in range(0, len(samples), batch_size):
            batch_samples = samples[offset: offset+batch_size]
            images, steering = get_images_and_labels(batch_samples, is_augment)

            yield shuffle(np.array(images), np.array(steering))
