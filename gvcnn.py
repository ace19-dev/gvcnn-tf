from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np

from slim.nets import inception_v2
from nets import googLeNet

slim = tf.contrib.slim


'''
CNN is the same as GoogLeNet.
Inception. v1. (a.k.a GoogLeNet)
'''
# def CNN(inputs, scope):
#     """
#         Final View Descriptors Generation
#
#         Extract the raw view descriptors. Compared with deeper CNN,
#         shallow FCN could have more position information, which is needed for
#         the followed grouping module and the deeper CNN will have the content information
#         which could represent the view feature better.
#
#         Args:
#         inputs: N x V x H x W x C tensor
#         scope:
#
#         Returns:
#         tensor_outs: output tensor corresponding to the final_endpoint.
#         end_points: a set of activations for external use, for example summaries or
#                     losses.
#
#         """
#
#     n_views = inputs.get_shape().as_list()[1]
#
#     # transpose views: (NxVxHxWxC) -> (VxNxHxWxC)
#     views = tf.transpose(inputs, perm=[1, 0, 2, 3, 4])
#
#     final_view_descriptors = []  # Final View Descriptors
#     for i in range(n_views):
#         batch_view = tf.gather(views, i)  # N x H x W x C
#
#         net, end_points = inception_v2.inception_v2_base(batch_view,
#                                                          scope=scope,
#                                                          use_separable_conv=True)
#
#         final_view_descriptors.append(net)
#
#     return final_view_descriptors, end_points


def grouping_scheme(num_group, view_discrimination_scores):
    group = {}

    g0 = tf.constant(0, dtype=tf.float32)
    g1 = tf.constant(1/num_group, dtype=tf.float32)
    g2 = tf.constant(2/num_group, dtype=tf.float32)
    g3 = tf.constant(3/num_group, dtype=tf.float32)
    g4 = tf.constant(4/num_group, dtype=tf.float32)
    g5 = tf.constant(5/num_group, dtype=tf.float32)

    for view_idx, score in enumerate(view_discrimination_scores):
        group_idx = tf.case(
            pred_fn_pairs=[
                (tf.logical_and(tf.greater_equal(score, g0), tf.less(score, g1)), lambda: tf.constant(1)),
                (tf.logical_and(tf.greater_equal(score, g1), tf.less(score, g2)), lambda: tf.constant(2)),
                (tf.logical_and(tf.greater_equal(score, g2), tf.less(score, g3)), lambda: tf.constant(3)),
                (tf.logical_and(tf.greater_equal(score, g3), tf.less(score, g4)), lambda: tf.constant(4)),
                (tf.logical_and(tf.greater_equal(score, g4), tf.less(score, g5)), lambda: tf.constant(5))],
            default=lambda: tf.constant(-1),
            exclusive=False)

        group[view_idx + 1] = group_idx

    return group


'''
1. 같은 그룹에 속하는 net 를 tf.div(tf.add(a, b), num_same_group) ?
2. 2d pooling ?
'''
def view_pooling(final_view_descriptors, group):

    '''
    :param final_view_descriptors:
    :param group: Grouping Scheme
    :return:
    '''

    # for i in range(NUM_SUB_RANGE):
    for i, view_desc in enumerate(final_view_descriptors):
        g0 = tf.cond(tf.equal(group[i], tf.Variable(0)),
                lambda: tf.add(view_desc, g0),
                lambda: tf.zeros(view_desc.shape))
        g1 = tf.cond(tf.equal(group[i], tf.Variable(1)),
                     lambda: tf.add(view_desc, g1),
                     lambda: tf.zeros(view_desc.shape))
        g2 = tf.cond(tf.equal(group[i], tf.Variable(2)),
                     lambda: tf.add(view_desc, g2),
                     lambda: tf.zeros(view_desc.shape))
        g3 = tf.cond(tf.equal(group[i], tf.Variable(3)),
                     lambda: tf.add(view_desc, g3),
                     lambda: tf.zeros(view_desc.shape))
        g4 = tf.cond(tf.equal(group[i], tf.Variable(4)),
                     lambda: tf.add(view_desc, g4),
                     lambda: tf.zeros(view_desc.shape))

    tf.logging.info("=========")


# def group_fusion():
#     return


def make_grouping_module(inputs,
                         num_group,
                         is_training=True,
                         dropout_keep_prob=0.8,
                         reuse=tf.AUTO_REUSE,
                         scope='InceptionV2',
                         global_pool=True):

    """
    Raw View Descriptor Generation

    first part of the network (FCN) to get the raw descriptor in the view level.

    TODO: The “FCN” part is the top five convolutional layers of GoogLeNet. (mid-level representation ??)

    Extract the raw view descriptors. Compared with deeper CNN,
    shallow FCN could have more position information, which is needed for
    the followed grouping module and the deeper CNN will have the content information
    which could represent the view feature better.

    Args:
    inputs: N x V x H x W x C tensor
    scope:
    """

    input_views = []
    raw_view_descriptors = []  # Raw View Descriptors

    # FC layer to obtain the discrimination scores from raw view descriptors
    view_discrimination_scores = []

    n_views = inputs.get_shape().as_list()[1]
    # transpose views: (NxVxHxWxC) -> (VxNxHxWxC)
    views = tf.transpose(inputs, perm=[1, 0, 2, 3, 4])

    with tf.variable_scope(scope, 'InceptionV2', [inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            for i in range(n_views):
                batch_view = tf.gather(views, i)  # N x H x W x C
                # FCN
                net, end_points = \
                    inception_v2.inception_v2_base(batch_view, scope=scope)

                input_views.append(batch_view)

                # The grouping module aims to learn the group information
                # to assist in mining the relationship among views.
                with tf.variable_scope('Logits'):
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

                    # 1 x 1 x 1024
                    net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b')
                    net = slim.flatten(net)
                    # net = slim.fully_connected(net, 512)
                    logits = slim.fully_connected(net, 1, activation_fn=None)
                    end_points['Logits'] = logits

                    # 각각의 point of view 에 해당하는 batch size input 에 대해서
                    # 평균 score 로 표시한다.
                    score = tf.nn.sigmoid(tf.log(tf.abs(logits)))
                    score = tf.reduce_mean(score, axis=0)
                    score = tf.reshape(score, [])

                    view_discrimination_scores.append(score)

    # grouping weight/scheme
    group_scheme = grouping_scheme(num_group, view_discrimination_scores)

    # TODO: how to get grouping weight ?
    group_weight = tf.zeros([2,2])

    return view_discrimination_scores, group_scheme, group_weight


def gvcnn(inputs,
          group_scheme,
          is_training=True,
          dropout_keep_prob=0.8,
          reuse=tf.AUTO_REUSE,
          scope='InceptionV2',
          global_pool=True):
    '''
    The second part of the network (CNN) and the group module, are used to extract
    the final view descriptors together with the discrimination scores, separately.
    '''
    input_views = []
    final_view_descriptors = []  # Final View Descriptors

    n_views = inputs.get_shape().as_list()[1]
    # transpose views: (NxVxHxWxC) -> (VxNxHxWxC)
    views = tf.transpose(inputs, perm=[1, 0, 2, 3, 4])

    with tf.variable_scope(scope, 'InceptionV2', [inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            for i in range(n_views):
                batch_view = tf.gather(views, i)  # N x H x W x C

                net, end_points = \
                    inception_v2.inception_v2_base(batch_view, scope=scope)

                input_views.append(batch_view)

                with tf.variable_scope('Logits'):
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

                    # 1 x 1 x 1024
                    net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
                    # logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                    #                      normalizer_fn=None, scope='Conv2d_1c_1x1')
                    # if spatial_squeeze:
                    #     logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
                end_points['Logits'] = net

                # TODO: Final View Descriptors + group_scheme <= View Pooling
