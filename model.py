from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np

import model

from slim.nets import inception_v2

slim = tf.contrib.slim


def FCN(inputs, scope):
    """
    Raw View Descriptor Generation

    Extract the raw view descriptors. Compared with deeper CNN,
    shallow FCN could have more position information,
    which is needed for the followed grouping module.
    and the deeper CNN will have the content information which could
    represent the view feature better.

    Args:
    inputs: N x V x H x W x C tensor

    Returns:
    tensor_outs: list of output tensor corresponding to the final_endpoint.
    # end_points: a set of activations for external use, for example summaries or
    #             losses.

    """

    n_views = inputs.get_shape().as_list()[1]

    # transpose views: (NxVxHxWxC) -> (VxNxHxWxC)
    views = tf.transpose(inputs, perm=[1, 0, 2, 3, 4])

    raw_view_descriptors = []  # Raw View Descriptors
    for i in range(n_views):

        view = tf.gather(views, i)  # NxWxHxC

        net, end_points = inception_v2.inception_v2_base(view,
                                                         final_endpoint='Mixed_5c',
                                                         min_depth=16,
                                                         depth_multiplier=1.0,
                                                         use_separable_conv=True,
                                                         data_format='NHWC',
                                                         scope=scope)
        raw_view_descriptors.append(net)

    return raw_view_descriptors, end_points



def grouping_module(raw_view_descriptors,
                    end_points,
                    num_classes,
                    reuse=None,
                    scope='InceptionV2',
                    global_pool=True,
                    # prediction_fn=tf.nn.sigmoid,
                    spatial_squeeze=True,
                    dropout_keep_prob=0.8):
    """
    The grouping module aims to learn the group information
    to assist in mining the relationship among views.

    :return:
    """

    # with tf.variable_scope('Logits'):
    discrimination_scores = []
    for i, net in enumerate(raw_view_descriptors):

        if global_pool:
            # Global average pooling.
            net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
            end_points['global_pool'] = net
        else:
            # Pooling with a fixed kernel size.
            kernel_size = inception_v2._reduced_kernel_size_for_small_input(net, [7, 7])
            net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                  scope='AvgPool_1a_{}x{}'.format(*kernel_size))
            end_points['AvgPool_1a'] = net
        # if not num_classes:
        #     return net, end_points

        # 1 x 1 x 1024
        net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
        logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                             normalizer_fn=None, scope='Conv2d_1c_1x1')
        if spatial_squeeze:
            logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
        end_points['Logits'] = logits
        # end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

        score = tf.nn.sigmoid(tf.log(tf.abs(net)))
        # predictions = tf.argmax(logits, 1)
        discrimination_scores.append(score)

    return discrimination_scores


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
          reuse=None,
          is_training=True,
          scope='gvcnn',
          global_pool=True,
          spatial_squeeze=True,
          dropout_keep_prob=0.8):

    """
    Args:
    inputs: a tensor of shape [batch_size, views, height, width, channels].
    """

    with tf.variable_scope(scope, 'gvcnn', [inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            raw_view_descriptors, end_points = model.FCN(inputs, scope)
            discrimination_scores = model.grouping_module(raw_view_descriptors,
                                                          end_points,
                                                          num_classes,
                                                          reuse,
                                                          scope)


    # Final View Descriptors -> View Pooling

    return discrimination_scores