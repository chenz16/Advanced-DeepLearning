# Semantic Segmentation
### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder
 
### Go through the rubrics

#### Ensure you've passed all the unit tests 

Unit tests passed 

#### Does the project load the pretrained vgg model?

The project is able to load the vgg model through the helper function, which was provided by Udacity 

#### Does the project learn the correct features from the images?

First, i took the output of vgg layer 7 as the input my first up-sampling layer. I tried to have an additional 1 by 1 layer before i do up-sampling. However, the results looks worse. I did the first up-sampling through convolution transpose operation, whose output depth was defined as "4*num_classes". Alternatively, the depth of layer could be set as any other number e.g. num_classess. This layer is named as Tconv_1;


	# 1X1 CONV WITH INPUT FROM PREVIOUS ENCODER
	#Encoder_out = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, 1)
	Encoder_out = vgg_layer7_out;
	# UPSAMPLIG - FIRST TRANSPOSE CONV LAYER

	Tconv_1     = tf.layers.conv2d_transpose(Encoder_out, 4*num_classes, 4, 2, padding = 'SAME',
		    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
		    bias_initializer=tf.zeros_initializer())


Then i used tf.add function to combine the vgg layer 4 output with the layer Tconv_1. The output of this layer is named as Tconv_2 with depth of 2*num_classes. tf.add requires the same depth of two. In order to make the two layers compatible, the vgg layer 4 ouput is pre-processed by 1x1 convolution before adding operation.

	layer_skip  = tf.layers.conv2d(vgg_layer4_out, 4*num_classes, 1, 1)
	skip_conv = tf.add(Tconv_1, layer_skip)
	Tconv_2     = tf.layers.conv2d_transpose(skip_conv, 2*num_classes, 4, 2, padding = 'SAME',
		  kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
		  bias_initializer=tf.zeros_initializer())

This process is repeated to combine the vgg layer 3 output with the layer Tconv_2. The output of this layer is named as Tconv_2 with the depth of num_classes to match the number of output categories. 

	layer_skip2  = tf.layers.conv2d(vgg_layer3_out, 2*num_classes, 1, 1)
	skip_conv2 = tf.add(Tconv_2, layer_skip2)
	Tconv_3     = tf.layers.conv2d_transpose(skip_conv2, num_classes, 16, 8, padding = 'SAME',
		  kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
		  bias_initializer=tf.zeros_initializer())

#### Does the project optimize the neural network?

yes

#### Does the project train the neural network?

yes, see the following training loss

#### Does the project train the model correctly?

yes, on average, the model decreases loss over time.

	Epoch 1 of 20: Training loss: 0.6516
	Epoch 2 of 20: Training loss: 0.5350
	Epoch 3 of 20: Training loss: 0.4866
	Epoch 4 of 20: Training loss: 0.3040
	Epoch 5 of 20: Training loss: 0.2892
	Epoch 6 of 20: Training loss: 0.1851
	Epoch 7 of 20: Training loss: 0.2925
	Epoch 8 of 20: Training loss: 0.6593
	Epoch 9 of 20: Training loss: 0.1850
	Epoch 10 of 20: Training loss: 0.3585
	Epoch 11 of 20: Training loss: 0.1523
	Epoch 12 of 20: Training loss: 0.1214
	Epoch 13 of 20: Training loss: 0.1383
	Epoch 14 of 20: Training loss: 0.1904
	Epoch 15 of 20: Training loss: 0.1477
	Epoch 16 of 20: Training loss: 0.1261
	Epoch 17 of 20: Training loss: 0.1110
	Epoch 18 of 20: Training loss: 0.1531
	Epoch 19 of 20: Training loss: 0.1330
	Epoch 20 of 20: Training loss: 0.1142

 
#### Does the project use reasonable hyperparameters?

The parameters are defined as:

	   epochs = 20
	   batch_size = 4
	   learning_rate = 0.001
	   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3)



#### Does the project correctly label the road?

The model can predict correctly for most of images. 
