# Traffic Sign Recognition 

The steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/1.png "Traffic Sign 1"
[image5]: ./examples/2.png "Traffic Sign 2"
[image6]: ./examples/3.png "Traffic Sign 3"
[image7]: ./examples/4.png "Traffic Sign 4"
[image8]: ./examples/5.png "Traffic Sign 5"
[image9]: ./examples/6.png "Traffic Sign 6"
[image10]: ./examples/7.png "Traffic Sign 7"
[image11]: ./examples/8.png "Traffic Sign 8"


### Data Set Summary & Exploration

#### 1. Basic summary of the data set.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distributes.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. How I preprocessed the image data. 

As a first step, I decided to convert the images to grayscale because grayscale represents the intensity of image, it is better for the network to process features(gradients), also grayscale has less dims, which makes the network work faster.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because normalized data is easier and faster to constrict.

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


#### 2. My final model architecture.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution1 3x3     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, VALID padding, outputs 14x14x6 	|
| Convolution2 3x3	    | 1x1 stride, VALID padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, VALID padding, outputs 5x5x6 		|
| Flatten				| outputs 5x5x6=400								|
| Fully connected1		| outputs 120									|
| RELU					|												|
| Dropout				| keep_prob 0.5									|
| Fully connected2		| outputs 84  									|
| RELU					|												|
| Dropout				| keep_prob 0.5									|
| Fully connected3		| outputs 43  									|
 


#### 3. Model training.

To train the model, I used an AdamOptimizer with batch_size=128, epoches=50 and learning_rate=0.001.

#### 4. The approach taken for finding a solution and getting the validation set accuracy.

My final model results were:
* validation set accuracy of 0.957 
* test set accuracy of 0.939

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. I Choose eight German traffic signs found on the web.

Here are eight German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image7]
![alt text][image8] ![alt text][image9] ![alt text][image10] ![alt text][image11]


#### 2. The predictions on these new traffic signs.

Here are the results of the prediction:

| Image									|Prediction	   							| 
|:---------------------:				|:-------------------------------------:| 
| Right-of-way at the next intersection	| Right-of-way at the next intersection	| 
| Turn left ahead     					| Turn left ahead 						|
| Keep right							| Keep right							|
| Priority road							| Priority road			 				|
| Speed limit (60km/h)					| Speed limit (60km/h)     				|
| General caution						| General caution      					|
| Road work								| Road work     						|
| Speed limit (30km/h)					| Speed limit (30km/h) 					|


The model was able to correctly guess all signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93.9%.


