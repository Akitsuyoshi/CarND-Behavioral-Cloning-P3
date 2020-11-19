# Import modules
import csv
import cv2
import numpy as np

# ../../../opt/carnd_p3/data/driving_log.csv
lines = []
with open('../../../opt/carnd_p3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
steering_measurements = []

for line in lines[1:]:
    source_path = line[0]
    center_filename = source_path.split('/')[-1]
    current_path = '../../../opt/carnd_p3/data/IMG/' + center_filename
    img = cv2.imread(current_path)
    img[:, :, (2, 1, 0)] = 0  # Convert from BGR to RGB
    images.append(img)
    steering_measurements.append(float(line[3]))


# Get features and labels
X_train = np.array(images)
y_train = np.array(steering_measurements)


# Import modeuls
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda

# Here is the definition of model
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Flatten())
model.add(Dense(1))
# Set optimizer and loss function
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# Train model
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3, verbose = 1)

# Save model
model.save('model.h5')
