# **Behavioral Cloning** 

[//]: # (Image References)
[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

### Goals
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report as per the [rubric](https://review.udacity.com/#!/rubrics/432/view).

### Submissions
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network 
* `writeup_report.md` summarizing the results

### Model Architecture and Training Strategy

#### 1. Model Architecture
I have used Nvidia model comprising of 4 convolutional layers with rectified linear unit (relu) as activation functions, followed by the flatten operation, and then by 4 dense layers. The model includes a dropout layer as well in order to prevent overfitting.

The input image has dimensions (160,320,3). This is cropped by 50px at the top and 25px at the bottom to only focus on the area of interest, resulting in an image of size (85,320,3).

```python
model.add(Cropping2D(cropping=((50,25), (0,0)), input_shape=(160,320,3)))
```

This is then normalized, which updates the values from the range [0,255] to [-0.5,0.5].
```python
model.add(Lambda(lambda x: x/256 - 0.5))
```

The normalized image is then run through a sequence of convolution, dropout, flatten and densely connected layers. The model is then compiled using `adam` optimizer for parameter tuning, and `mean square error` loss calculation strategy.

```python
model.compile(loss='mse', optimizer='adam')
```

Further detail on the model/architecture in the code and summary below:

**Code:**

```python
model = Sequential()
    model.add(Cropping2D(cropping=((50,25), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: x/256 - 0.5))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', activation="relu"))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation="relu"))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation="relu"))
    model.add(Dropout(keep_prob))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
```

**Summary**

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
cropping2d_1 (Cropping2D)    (None, 85, 320, 3)        0         
_________________________________________________________________
lambda_1 (Lambda)            (None, 85, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 41, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 19, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 6, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 4, 33, 64)         36928     
_________________________________________________________________
dropout_1 (Dropout)          (None, 4, 33, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 8448)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               844900    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 981,819
Trainable params: 981,819
Non-trainable params: 0
_________________________________________________________________
```

#### 2. Training Strategy
* I generated the training image dataset by recording my session (around 5 laps) around Track 1. I did a mix of:
	* couple of laps of driving in the center of the lane.
	* going off the center and recover in straight roads and turns.
	* driving close to the left of the lane in case of left turns and vice versa so that the car won't offshoot at the turns.
* Train:Test split of 80:20
* Tested for various number of epochs, and check when the loss would plateau. Possibly because I had a good amount of training data, my losses were pretty good at the end of epoch 1 or 2 itself, and started to overfit after that.
* I used the images from all 3 cameras. Also, since steering in track #1 is biased towards left, I used the flipped images as well to compensate for that.

After several iterations and tuning, the car was able to drive autonomously. Here is the video of the car doing several laps around the lake:

[![Self Driving Car - Behavioral Cloning](https://i.ytimg.com/vi/HNsagL1y0-k/hqdefault.jpg)](https://www.youtube.com/watch?v=HNsagL1y0-k&feature=youtu.be)

