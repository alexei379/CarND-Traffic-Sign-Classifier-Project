# Traffic Sign Classification Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goals / steps of this project are the following:
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

Here is an exploratory visualization of the data set. It is a bar chart showing that classes have the same distribution in trainig, validation and test data sets.
![](https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/report_images/distrib_all_pre_random.png)

But the number of training images is not evenly distributed between classes:
![](https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/report_images/distrib_train_pre_random.png)

## Design and Test a Model Architecture

### 1. Preprocessing trainig images

First I generate fake random training data by rotating ±15°, shifting horizontally and vertically ±2 pixels. This allows to get more trainig data for better network trainig.
To make the distribution of training data between classes I generate different number of random images for each class.

![](https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/report_images/distrib_train_after_random.png)

After this images are converted to grayscale as LeNet architecture works with grayscale images.
As the last step I normalize the image data to make it easier for the optimizer to find an optimal solution. 

Sample images from the preprocessed trainig set:
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

I choose to use LeNet-5 architecture from the [CarND-LeNet-Lab](https://github.com/udacity/CarND-LeNet-Lab) as a base model. It worked well on recognizing complex features of hand-written digits so I decided to give it a try. As we have more classes to classify, I added extra fully connected layer to gather more feature information in it. To prevent overfitting I also added a dropout layer. 

Also I was iteratively adjusting parameters like learning rate, number of epochs and keep probablity to get the final results. I decided to add a "break" from the epochs cycle once the target accuracy is reached.

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
| ![](https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/test_signs/13.jpg) | Yield	|
 Yield	|| ![](https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/test_signs/15.jpg) | No vehicles |
| ![](https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/test_signs/17.jpg) | No entry | No entry |
| ![](https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/test_signs/18.jpg) | General caution |
 General caution || ![](https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/test_signs/25.jpg) | Road work |
| ![](https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/test_signs/33.jpg) | Turn right ahead |
 Turn right ahead || ![](https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/test_signs/35.jpg) | Ahead only | Ahead only |
| ![](https://raw.githubusercontent.com/alexei379/CarND-Traffic-Sign-Classifier-Project/master/test_signs/36.jpg) | Go straight or right | Go straight or right |

The model was able to correctly guess 10 of the 10 traffic signs, which gives an accuracy of 100%. The accuracy is in line with the test data set accuracy of 94.5%. 
I believe that such high accuracy was due to images on Google Street View are collected in a similar way as the trainig data set was created (i.e. pictures of real signs in the streets).

### 3. Model Certainty - Softmax Probabilities


Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| File | Top Softmax Probabilities |
| 1 | 2 | 3 | 4 | 5 |
| **1.jpg
Speed limit (30km/h)** | Speed limit (30km/h)
0.999918222427 | Speed limit (50km/h)
0.000050964361 | Speed limit (70km/h)
0.000028002120 | Speed limit (80km/h)
0.000002151676 | Speed limit (20km/h)
0.000000517173 |
| **12.jpg
Priority road** | Priority road
0.999999642372 | Roundabout mandatory
0.000000234978 | Road work
0.000000132352 | Keep right
0.000000041225 | Turn left ahead
0.000000024632 |
| **13.jpg
Yield** | Yield
1.000000000000 | Priority road
0.000000000000 | No vehicles
0.000000000000 | No passing
0.000000000000 | Go straight or right
0.000000000000 |
| **15.jpg
No vehicles** | No vehicles
0.990516245365 | Speed limit (70km/h)
0.005945517682 | Speed limit (50km/h)
0.003034086665 | Yield
0.000159200528 | Priority road
0.000144167629 |
| **17.jpg
No entry** | No entry
0.999993562698 | Turn left ahead
0.000003147254 | Stop
0.000002419770 | Turn right ahead
0.000000375887 | Keep left
0.000000311693 |
| **18.jpg
General caution** | General caution
1.000000000000 | Pedestrians
0.000000041053 | Traffic signals
0.000000021591 | Keep right
0.000000000019 | Go straight or left
0.000000000001 |
| **25.jpg
Road work** | Road work
0.940166592598 | Children crossing
0.017493918538 | Slippery road
0.015498344786 | Pedestrians
0.006842988543 | Priority road
0.006191547960 |
| **33.jpg
Turn right ahead** | Turn right ahead
1.000000000000 | Ahead only
0.000000000590 | Priority road
0.000000000014 | Keep left
0.000000000003 | Turn left ahead
0.000000000001 |
| **35.jpg
Ahead only** | Ahead only
0.999990344048 | Turn right ahead
0.000009497156 | Road work
0.000000108985 | Right-of-way at the next intersection
0.000000014637 | Priority road
0.000000004412 |
| **36.jpg
Go straight or right** | Go straight or right
0.999999642372 | Ahead only
0.000000155470 | Keep right
0.000000065167 | Turn left ahead
0.000000043115 | Turn right ahead
0.000000007037 |
