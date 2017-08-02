# Traffic Sign Classification Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goals/steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
## Writeup

You're reading it! And here is a link to my project code [[ipynb]](https://github.com/alexei379/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) [[html]](https://htmlpreview.github.io/?https://github.com/alexei379/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.html)

---
## Data Set Summary & Exploration

### 1. Basic summary of the data set
We'll be using [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) for this project. Data set comes in a form of pickled files. The pickled data dictionaries of interest:
'features' is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
'labels' is a 1D array containing the label/class id of the traffic sign.
The file signnames.csv contains id -> name mappings for each id.

I used Python to calculate summary statistics of the traffic signs data set:

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

### 2. Visualization of the dataset

Here is an exploratory visualization of the data set. It is a bar chart showing that classes have the same distribution in training, validation and test data sets.
![](https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/report_images/distrib_all_pre_random.png)

But the number of training images is not evenly distributed between classes:
![](https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/report_images/distrib_train_pre_random.png)

## Design and Test a Model Architecture

### 1. Preprocessing training images

First I generate fake random training data by rotating ±15°, shifting horizontally and vertically ±2 pixels. This allows getting more training data for better network training. To make the distribution of training data between classes I generate a different number of random images for each class.

![](https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/report_images/distrib_train_after_random.png)

After this images are converted to grayscale as LeNet architecture works with grayscale images.
As the last step, I normalize the image data to make it easier for the optimizer to find an optimal solution. 

Sample images from the preprocessed training set:
![](https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/report_images/train_images_samples.png)

### 2. Model architecture

My final model is a convolutional network based on LeNet-5 architecture. 
It consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 1x1     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|	Activation											|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 1x1     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|	Activation											|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				| 
| Flatten | 5x5x16 -> 400x1 |
| Fully connected		| 400 -> 250	|
| RELU					|	Activation											|
| Dropout		| Keep probability = 0.25	|
| Fully connected		| 250 -> 160	|
| RELU					|	Activation											|
| Fully connected		| 160 -> 84	|
| RELU					|	Activation											|
| Fully connected		| 84 -> 43	|
| Output				| 43 logits        									|

### 3. Model training

To train the model I used and optimizer that implements the Adam algorithm to minimize the loss - mean of softmax cross entropy between logits and labels.

* Learning rate = 0.001
* Number of epochs = 100 
* Batch size = 128
* Keep probability = 0.25

### 4. Solution Approach

I choose to use LeNet-5 architecture from the [CarND-LeNet-Lab](https://github.com/udacity/CarND-LeNet-Lab) as a base model. It worked well on recognizing complex features of handwritten digits so I decided to give it a try. As we have more classes to classify, I added an extra fully connected layer to gather more feature information in it. To prevent overfitting I also added a drop out layer.

Also, I was iteratively adjusting parameters like learning rate, the number of epochs and keep probability to get the final results. I decided to add a "break" from the epochs cycle once the target accuracy is reached.

My final model results were:
* training set accuracy of 0.986
* validation set accuracy of 0.967 
* test set accuracy of 0.945

Accuracy over 0.93 on all sets proves that the chosen approach successfully solves the problem. 

## Test a Model on New Images

### 1. Ten German traffic signs

I used [Google Street View](https://www.google.com/maps/place/Bremen,+Germany/) to get ten signs from Bremen streets and used my model to predict the traffic sign type.

| Image from the web         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| ![](https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/test_signs/1.jpg) | Speed limit |
| ![](https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/test_signs/12.jpg) | Priority road |
| ![](https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/test_signs/13.jpg) | Yield	|
| ![](https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/test_signs/15.jpg) | No vehicles |
| ![](https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/test_signs/17.jpg) | No entry |
| ![](https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/test_signs/18.jpg) | General caution |
| ![](https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/test_signs/25.jpg) | Road work |
| ![](https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/test_signs/33.jpg) | Turn right ahead |
| ![](https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/test_signs/35.jpg) | Ahead only |
| ![](https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/test_signs/36.jpg) | Go straight or right |


### 2. Performance on New Images

Here are the results of the prediction:

| Image	| Expected	| Prediction | 
|:-----:|:--------:|:----------:| 
| ![](https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/test_signs/1.jpg) | Speed limit | Speed limit |
| ![](https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/test_signs/12.jpg) | Priority road | Priority road |
| ![](https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/test_signs/13.jpg) | Yield	| Yield	|
| ![](https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/test_signs/15.jpg) | No vehicles | No vehicles |
| ![](https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/test_signs/17.jpg) | No entry | No entry |
| ![](https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/test_signs/18.jpg) | General caution | General caution |
| ![](https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/test_signs/25.jpg) | Road work | Road work |
| ![](https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/test_signs/33.jpg) | Turn right ahead | Turn right ahead |
| ![](https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/test_signs/35.jpg) | Ahead only | Ahead only |
| ![](https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/test_signs/36.jpg) | Go straight or right | Go straight or right |

The model was able to correctly guess 10 of the 10 traffic signs, which gives an accuracy of 100%. The accuracy is in line with the test data set accuracy of 94.5%. 
I believe that such high accuracy was due to images on Google Street View are collected in a similar way as the training data set was created (i.e. pictures of real signs in the streets).

### 3. Model Certainty - Softmax Probabilities

The model is pretty certain about all images. The lowest "first" probability is for "Road work" sign is 0.940166592598. 
The model recognizes some features of "Children crossing" and "Slippery road" signs, which is correct as they are also represented by a triangle with a symbol in the middle.

Table with top-5 softmax probabilities (left to right: most certain -> least certain)
<table>
  <tbody>
    <tr>
      <th rowspan="2">Image</th>
      <th colspan="5">Top Softmax Probabilities</th>
    </tr>
    <tr>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
    <tr>
      <td><img src="https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/test_signs/1.jpg"/><br>
      Speed limit (30km/h)</td>
      <td>Speed limit (30km/h)<br>
      0.999918222427</td>
      <td>Speed limit (50km/h)<br>
      0.000050964361</td>
      <td>Speed limit (70km/h)<br>
      0.000028002120</td>
      <td>Speed limit (80km/h)<br>
      0.000002151676</td>
      <td>Speed limit (20km/h)<br>
      0.000000517173</td>
    </tr>
    <tr>
      <td><img src="https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/test_signs/12.jpg"/><br>
      Priority road</td>
      <td>Priority road<br>
      0.999999642372</td>
      <td>Roundabout mandatory<br>
      0.000000234978</td>
      <td>Road work<br>
      0.000000132352</td>
      <td>Keep right<br>
      0.000000041225</td>
      <td>Turn left ahead<br>
      0.000000024632</td>
    </tr>
    <tr>    
      <td><img src="https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/test_signs/13.jpg"/><br>
      Yield</td>
      <td>Yield<br>
      1.000000000000</td>
      <td>Priority road<br>
      0.000000000000</td>
      <td>No vehicles<br>
      0.000000000000</td>
      <td>No passing<br>
      0.000000000000</td>
      <td>Go straight or right<br>
      0.000000000000</td>
    </tr>
    <tr>
      <td><img src="https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/test_signs/15.jpg"/><br>
      No vehicles</td>
      <td>No vehicles<br>
      0.990516245365</td>
      <td>Speed limit (70km/h)<br>
      0.005945517682</td>
      <td>Speed limit (50km/h)<br>
      0.003034086665</td>
      <td>Yield<br>
      0.000159200528</td>
      <td>Priority road<br>
      0.000144167629</td>
    </tr>
    <tr>
      <td><img src="https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/test_signs/17.jpg"/><br>
      No entry</td>
      <td>No entry<br>
      0.999993562698</td>
      <td>Turn left ahead<br>
      0.000003147254</td>
      <td>Stop<br>
      0.000002419770</td>
      <td>Turn right ahead<br>
      0.000000375887</td>
      <td>Keep left<br>
      0.000000311693</td>
    </tr>
    <tr>
      <td><img src="https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/test_signs/18.jpg"/><br>
      General caution</td>
      <td>General caution<br>
      1.000000000000</td>
      <td>Pedestrians<br>
      0.000000041053</td>
      <td>Traffic signals<br>
      0.000000021591</td>
      <td>Keep right<br>
      0.000000000019</td>
      <td>Go straight or left<br>
      0.000000000001</td>
    </tr>
    <tr>
      <td><img src="https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/test_signs/25.jpg"/><br>
      Road work</td>
      <td>Road work<br>
      0.940166592598</td>
      <td>Children crossing<br>
      0.017493918538</td>
      <td>Slippery road<br>
      0.015498344786</td>
      <td>Pedestrians<br>
      0.006842988543</td>
      <td>Priority road<br>
      0.006191547960</td>
    </tr>
    <tr>
      <td><img src="https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/test_signs/33.jpg"/><br>
      Turn right ahead</td>
      <td>Turn right ahead<br>
      1.000000000000</td>
      <td>Ahead only<br>
      0.000000000590</td>
      <td>Priority road<br>
      0.000000000014</td>
      <td>Keep left<br>
      0.000000000003</td>
      <td>Turn left ahead<br>
      0.000000000001</td>
    </tr>
    <tr>
      <td><img src="https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/test_signs/35.jpg"/><br>
      Ahead only</td>
      <td>Ahead only<br>
      0.999990344048</td>
      <td>Turn right ahead<br>
      0.000009497156</td>
      <td>Road work<br>
      0.000000108985</td>
      <td>Right-of-way at the next intersection<br>
      0.000000014637</td>
      <td>Priority road<br>
      0.000000004412</td>
    </tr>
    <tr>
      <td><img src="https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/test_signs/36.jpg"/><br>
      Go straight or right</td>
      <td>Go straight or right<br>
      0.999999642372</td>
      <td>Ahead only<br>
      0.000000155470</td>
      <td>Keep right<br>
      0.000000065167</td>
      <td>Turn left ahead<br>
      0.000000043115</td>
      <td>Turn right ahead<br>
      0.000000007037</td>
    </tr>
  </tbody>
</table>
