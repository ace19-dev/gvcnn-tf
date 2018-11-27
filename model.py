from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np

from slim.nets import inception_v2

slim = tf.contrib.slim


def FCN(inputs):
    """
    Raw View Descriptor Generation

    Extract the raw view descriptors. Compared with deeper CNN,
    shallow FCN could have more position information,
    which is needed for the followed grouping module.
    and the deeper CNN will have the content information which could
    represent the view feature better.

    Args:
    inputs: list of a tensor of shape [[batch_size, height, width, channels], ...].

    Returns:
    tensor_outs: list of output tensor corresponding to the final_endpoint.
    # end_points: a set of activations for external use, for example summaries or
    #             losses.


    """

    fcn_results = []  # Raw View Descriptors
    for i, input in enumerate(inputs):

        net, _ = inception_v2.inception_v2_base(input,
                                          final_endpoint='Mixed_5c',
                                          min_depth=16,
                                          depth_multiplier=1.0,
                                          use_separable_conv=True,
                                          data_format='NHWC',
                                          scope=None)
        fcn_results.append(net)

    return fcn_results



def grouping_module(fcn_nets,
                    reuse=None,
                    scope='FC',
                    global_pool=True,
                    prediction_fn=tf.nn.sigmoid,
                    spatial_squeeze=True,
                    dropout_keep_prob=0.8):
    """
    The grouping module aims to learn the group information
    to assist in mining the relationship among views.

    :return:
    """

    discrimination_logits = []
    for i, net in enumerate(fcn_nets):

        # Final pooling and prediction
        with tf.variable_scope('Logits'):
            if global_pool:
                # Global average pooling.
                net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                # end_points['global_pool'] = net
            else:
                # Pooling with a fixed kernel size.
                kernel_size = inception_v2._reduced_kernel_size_for_small_input(net, [7, 7])
                net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                      scope='AvgPool_1a_{}x{}'.format(*kernel_size))
                # end_points['AvgPool_1a'] = net
            # if not num_classes:
            #     return net, end_points

        #     # 1 x 1 x 1024
        #     net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
        #     logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
        #                          normalizer_fn=None, scope='Conv2d_1c_1x1')
        #     if spatial_squeeze:
        #         logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
        # # end_points['Logits'] = logits
        # end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

            logits = tf.nn.sigmoid(tf.log(tf.abs(net)))
            # predictions = tf.argmax(logits, 1)
            discrimination_logits.append(logits)

    return discrimination_logits


# def view_pooling():
#
#
#
#
#
# def group_fusion():
#     return


def gvcnn(inputs,
          num_classes=1000,
          is_training=True,
          dropout_keep_prob=0.8,
          scope='gvcnn'):

    """
    Args:
    inputs: list of a tensor of shape [[batch_size, height, width, channels], ...].
    """

    fcn_results = []    # Raw View Descriptors
    for i, input in enumerate(inputs):
        # result = (logits, end_points)
        result = inception_v2.inception_v2(
            input,
            num_classes=num_classes,
            is_training=is_training,
            dropout_keep_prob=dropout_keep_prob,
            min_depth=16,
            depth_multiplier=1.0,
            prediction_fn=slim.softmax,
            spatial_squeeze=True,
            reuse=None,
            scope=scope,
            global_pool=False)

        fcn_results.append(result)

    # Final View Descriptors -> View Pooling