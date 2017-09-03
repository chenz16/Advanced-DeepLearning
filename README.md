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

In the "layers" function, I first used a 1x1 convolution operation to convert the outputs with depth of 'num_classes'.

Then i did the first up-sampling through convolution transpose operation, whose output depth was defined as "4*num_classes". Alternatively, the depth of layer could be set as any other number e.g. num_classess. This layer is named as Tconv_1;

Then i used tf.add function to combine the vgg layer 4 output with the layer Tconv_1. The output of this layer is named as Tconv_2 with depth of 2*num_classes. Unlike tf.stack, tf.add adds the layers, which requires the same depth of two. In order to make the two layers compatible, the vgg layer 4 ouput is pre-processed by 1x1 convolution before adding operation.

This process is repeated to combine the vgg layer 3 output with the layer Tconv_2. The output of this layer is named as Tconv_2 with the depth of num_classes to match the number of output categories. 

#### Does the project optimize the neural network?

yes

#### Does the project train the neural network?

yes, see the following training loss

#### Does the project train the model correctly?

yes, on average, the model decreases loss over time.

Epoch 1 of 40: Training loss: 0.6404
Epoch 2 of 40: Training loss: 0.5130
Epoch 3 of 40: Training loss: 0.4573
Epoch 4 of 40: Training loss: 0.3928
Epoch 5 of 40: Training loss: 0.2905
Epoch 6 of 40: Training loss: 0.3105
Epoch 7 of 40: Training loss: 0.2828
Epoch 8 of 40: Training loss: 0.1651
Epoch 9 of 40: Training loss: 0.2030
Epoch 10 of 40: Training loss: 0.1985
Epoch 11 of 40: Training loss: 0.1859
Epoch 12 of 40: Training loss: 0.2196
Epoch 13 of 40: Training loss: 0.2058
Epoch 14 of 40: Training loss: 0.1697
Epoch 15 of 40: Training loss: 0.0699
Epoch 16 of 40: Training loss: 0.1331
Epoch 17 of 40: Training loss: 0.0521
Epoch 18 of 40: Training loss: 0.0968
Epoch 19 of 40: Training loss: 0.1506
Epoch 20 of 40: Training loss: 0.0970
Epoch 21 of 40: Training loss: 0.2309
Epoch 22 of 40: Training loss: 0.0561
Epoch 23 of 40: Training loss: 0.2112
Epoch 24 of 40: Training loss: 0.0826
Epoch 25 of 40: Training loss: 0.0724
Epoch 26 of 40: Training loss: 0.0849
Epoch 27 of 40: Training loss: 0.0474
Epoch 28 of 40: Training loss: 0.0585
Epoch 29 of 40: Training loss: 0.0809
Epoch 30 of 40: Training loss: 0.0755
Epoch 31 of 40: Training loss: 0.0580
Epoch 32 of 40: Training loss: 0.0854
Epoch 33 of 40: Training loss: 0.0437
Epoch 34 of 40: Training loss: 0.0757
Epoch 35 of 40: Training loss: 0.0911
Epoch 36 of 40: Training loss: 0.0640
Epoch 37 of 40: Training loss: 0.0439
Epoch 38 of 40: Training loss: 0.0504
Epoch 39 of 40: Training loss: 0.0580
Epoch 40 of 40: Training loss: 0.0447
Training Finished. Saving test images to: ./runs/1504455239.3679957
 
#### Does the project use reasonable hyperparameters?

The parameters are defined as:

   epochs = 20
   batch_size = 4
   learning_rate = 0.001
   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3)



#### Does the project correctly label the road?

The model can predict correctly for most of images. 
