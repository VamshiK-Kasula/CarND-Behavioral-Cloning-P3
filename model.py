import csv
import cv2
import numpy as np
import os 
import sys
import glob

import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Conv2D
from keras.layers.pooling import MaxPooling2D

epochs = 4
batch_size = 64
steering_correction = [0.0, 0.2, -0.2]

def read_csv():
    os.chdir('/home/workspace/CarND-Behavioral-Cloning-P3/data_samples')
    lines = []
    for sub_dir in  next(os.walk('.'))[1]:
        with open(sub_dir + '/driving_log.csv') as csv_file:
            reader = csv.reader(csv_file)
            for line in reader:
                lines.append(line)
    return lines

def generator(data, batch_size):
    data_length = len(data)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(data)
        for offset in range(0, data_length, batch_size):
            batch_data = data[offset:offset+batch_size]
            images = []
            steering_values = []
            for batch_sample in batch_data:
                steering = float(batch_sample[3])
                for i in range(3):
                    image = cv2.imread(batch_sample[i].strip())
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(rgb_image)
                    steering_values.append(steering + steering_correction[i])
                    images.append(cv2.flip(image,1))
                    steering_values.append((steering + steering_correction[i]) * -1.0)
            
            X_train = np.array(images)
            y_train = np.array(steering_values)
            yield sklearn.utils.shuffle(X_train, y_train)


def create_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    model.add(Conv2D(24,(5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(36,(5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(48,(5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

if __name__ == '__main__':
    lines = read_csv()
    X_train, X_valid = train_test_split(lines, test_size=0.2)

    train_generator = generator(X_train, batch_size)
    valid_generator = generator(X_valid, batch_size)
    model = create_model()

    model.compile(loss='mse', optimizer='adam')
    # model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch = epochs)
    model.fit_generator(train_generator, steps_per_epoch = np.ceil(len(X_train)/batch_size), validation_data = valid_generator, validation_steps = np.ceil(len(X_valid)/batch_size),epochs = epochs, verbose = 1 )
    model.save("../model2.h5")