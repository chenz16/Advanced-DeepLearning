import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    #tf.saved_model.loader.load(sess,tags,export_dir,**saver_kwargs)
    tf.saved_model.loader.load(sess,[vgg_tag], vgg_path)
    graph = tf.get_default_graph();
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # 1X1 CONV WITH INPUT FROM PREVIOUS ENCODER
    #Encoder_out = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, 1)
    Encoder_out = vgg_layer7_out;
    # UPSAMPLIG - FIRST TRANSPOSE CONV LAYER

    Tconv_1     = tf.layers.conv2d_transpose(Encoder_out, 4*num_classes, 4, 2, padding = 'SAME',
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                    bias_initializer=tf.zeros_initializer())
    #tf.Print(Tconv_1, [tf.shape(Tconv_1)])


    # UPSAMPING - SECOND TRANSPOSE CONV LAYER, ADD SKIP LAYER FIRST
    # tf.Print(vgg_layer4_out, [tf.shape(vgg_layer4_out)])
    layer_skip  = tf.layers.conv2d(vgg_layer4_out, 4*num_classes, 1, 1)
    #layer_skip = vgg_layer4_out
    #tf.Print(layer_skip, [tf.shape(layer_skip)])

    #layer_skip = vgg_layer4_out
    skip_conv = tf.add(Tconv_1, layer_skip)
    #skip_conv = tf.stack([Tconv_1, layer_skip])
    Tconv_2     = tf.layers.conv2d_transpose(skip_conv, 2*num_classes, 4, 2, padding = 'SAME',
                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                  bias_initializer=tf.zeros_initializer())
    #tf.Print(Tconv_2, [tf.shape(Tconv_2)])

    # UPSAMPING - THRID TRANSPOSE CONV LAYER, ADD SKIP LAYER FIRST
    layer_skip2  = tf.layers.conv2d(vgg_layer3_out, 2*num_classes, 1, 1)
    #layer_skip2 = vgg_layer3_out
    skip_conv2 = tf.add(Tconv_2, layer_skip2)
    Tconv_3     = tf.layers.conv2d_transpose(skip_conv2, num_classes, 16, 8, padding = 'SAME',
                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                  bias_initializer=tf.zeros_initializer())
    #tf.Print(Tconv_3, [tf.shape(Tconv_3)])

    return Tconv_3
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    #RESHAPE
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    #print(logits.shape, "\n ", labels.shape)

    # DEFINE LOSS
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

    # DEFIINE OPTIMIZER
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Define train_op to minimise loss
    train_op = optimizer.minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    keep_prob_stat = 0.8
    learning_rate_stat = 0.001
    for epoch in range(epochs):
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: image,
                                          correct_label: label,
                                          keep_prob: keep_prob_stat,
                                          learning_rate:learning_rate_stat})
        print("Epoch %d of %d: Training loss: %.4f" %(epoch+1, epochs, loss))
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    epochs = 20
    batch_size = 4

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        correct_label = tf.placeholder(dtype=tf.float32, shape=(None, None, None, num_classes))

        learning_rate = tf.placeholder(dtype=tf.float32)
        logits, train_op, cross_entropy_loss = optimize(last_layer, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, image_input,
                 correct_label, keep_prob, learning_rate)
        saver = tf.train.Saver()
        saver.save(sess, 'checkpoints/model1.ckpt')
        saver.export_meta_graph('checkpoints/model1.meta')
        tf.train.write_graph(sess.graph_def, './checkpoints/', 'model1.pb', False)
        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
