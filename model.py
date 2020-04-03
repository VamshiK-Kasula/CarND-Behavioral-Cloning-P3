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

def read_csv(file_path):
    lines = []
    with open(file_path + '/driving_log.csv') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            lines.append(line)
    return lines

def read_data():
    os.chdir('/home/workspace/CarND-Behavioral-Cloning-P3/data_samples')
    for sub_dir in  next(os.walk('.'))[1]:
        lines = read_csv(sub_dir)
        images = []
        steering_values = []
        for line in lines[1:]:
            steering = float(line[3])
            for i in range(3):
                image = cv2.imread(line[i].strip())
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(rgb_image)
                steering_values.append(steering + steering_correction[i])
                images.append(cv2.flip(image,1))
                steering_values.append((steering + steering_correction[i]) * -1.0)

    return sklearn.utils.shuffle(np.array(images), np.array(steering_values))


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

def trainAndSave(model, inputs, outputs, modelFile, epochs = 3):
    model.compile(loss='mse', optimizer='adam')
    model.fit(inputs, outputs, validation_split=0.2, shuffle=True, nb_epoch=epochs)
    model.save(modelFile)

if __name__ == '__main__':
    model = create_model()
    X_train, y_train = read_data()
    X_valid, y_valid = read_data()
    model.compile(loss='mse', optimizer='adam')
    # model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch = epochs)
    model.fit_generator((X_train, y_train), steps_per_epoch = np.ceil(len(X_train)/batch_size), validation_data = (X_valid, y_valid), validation_steps = np.ceil(len(X_train)/batch_size),epochs = epochs, verbose = 1 )
    model.save("../model2.h5")
