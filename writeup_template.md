# **Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[center]: ./results/center.jpg "camera test image center"
[center_flipped]: ./results/center_flipped.jpg "camera test image center flipped"
[result]: ./results/result.png "result"

### Files Submitted & Code Quality

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided [simulator](https://github.com/udacity/self-driving-car-sim) and my [drive.py](https://github.com/VamshiK-Kasula/CarND-Behavioral-Cloning-P3) file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The [model.py](https://github.com/VamshiK-Kasula/CarND-Behavioral-Cloning-P3) file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

---
### Test runs and loading data

Along with the provided test data, two other test runs are performed. In one of the test run , ego car was driven in the opposite direction for robustness of data. Data from all the three test runs is loaded in a loop

```python
def read_csv():
    os.chdir('/home/workspace/CarND-Behavioral-Cloning-P3/data_samples')
    lines = []
    for sub_dir in  next(os.walk('.'))[1]:
        with open(sub_dir + '/driving_log.csv') as csv_file:
            reader = csv.reader(csv_file)
            for line in reader:
                lines.append(line)
    return lines
```
a generator is used to feed the test data to the model, the data read from the csv files is fed to the generator

```python
def generator(data, batch_size):
    ...
        yield sklearn.utils.shuffle(X_train, y_train)
```
All the three camera images (left, center and right) are read for the particular position of the ego and appended to the image data set.

```python              
steering_correction = [0.0, 0.2, -0.2]
for i in range(3):
    image = cv2.imread(batch_sample[i].strip())
    steering_values.append(steering + steering_correction[i])
```
Also the steering values of the ego are read and a correction factor of 0.2 is added to the left and 0.2 is subtracted to compute the left and right steering values. Each image is flipped and steering value is reversed to augment the data.

![alt text][center]
![alt text][center_flipped]

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Model summary is as follows:

The successful [NVIDIA CNN](https://arxiv.org/abs/1604.07316) model is adapted to drive the car along the path.
Input image size : 320x160
Number of image samples : 113,728
|Layer                            |Output Shape   |Activation |
|-------------------------------  |:-------------:|:---------:|
|lambda_1 (Lambda)                | 160, 320, 3   |-          |
|cropping2d_1 (Cropping2D)        | 90, 320, 3    |-          |
|convolution2d_1 (Convolution2D)  | 43, 158, 24   |ReLU       |
|convolution2d_2 (Convolution2D)  | 20, 77, 36    |ReLU       |
|convolution2d_3 (Convolution2D)  | 8, 37, 48     |ReLU       |
|convolution2d_4 (Convolution2D)  | 6, 35, 64     |ReLU       |
|convolution2d_5 (Convolution2D)  | 4, 33, 64     |ReLU       |
|flatten_1 (Flatten)              | 8448          |-          |
|dense_1 (Dense)                  | 100           |-          |
|dense_2 (Dense)                  | 50            |-          |
|dense_3 (Dense)                  | 10            |-          |
|dense_4 (Dense)                  | 1             |-          |


#### 3. Creation of the Training Set & Training Process

The sample data was split into 0.8 training data and 0.2 validation data sets.

```python
    X_train, X_valid = train_test_split(lines, test_size=0.2)

    train_generator = generator(X_train, batch_size)
    valid_generator = generator(X_valid, batch_size)
```
fit_generator function provided by Keras is used to train the model using the data generated batch-by-batch from train_generator and valid_generator

an epoch size of 4 and batch size of 64 is used

```python
    model.fit_generator(train_generator, steps_per_epoch = np.ceil(len(X_train)/batch_size), validation_data = valid_generator, validation_steps = np.ceil(len(X_valid)/batch_size),epochs = epochs, verbose = 1 )
```

The model trained is used to drive the car autonomously. The ego car perfectly maneuvered through the track by staying only on the drivable portion of the car. Recorded result is available in the following link.

[![alt text][center]](<https://www.youtube.com/watch?v=5q-4W3dad7Y>)