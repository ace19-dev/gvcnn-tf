from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np

from slim.nets import inception_v2
from nets import googLeNet

slim = tf.contrib.slim


def grouping(scores, num_group, num_views):
    g = np.full((num_group, num_views), False)

    for idx, s in enumerate(scores):
        if 0.0 <= s < 0.2:      # 0 group
            g[0, idx] = True
        elif 0.2 <= s < 0.4:    # 1
            g[1, idx] = True
        elif 0.4 <= s < 0.5:    # 2
            g[2, idx] = True
        elif 0.5 <= s < 0.6:    # 3
            g[3, idx] = True
        elif 0.6 <= s < 0.7:    # 4
            g[4, idx] = True
        elif 0.7 <= s < 0.8:    # 5
            g[5, idx] = True
        elif 0.8 <= s < 0.9:    # 6
            g[6, idx] = True
        elif 0.9 <= s < 1.0:    # 7 group
            g[7, idx] = True

    return g


# TODO: modified according to equation2 in paper correctly when totally understand.
# average value per group
def group_weight(scores, group_scheme):
    num_group = group_scheme.shape[0]
    num_views = group_scheme.shape[1]

    weight = np.zeros((num_group, 1), dtype=np.float32)

    nans = []
    for i in range(num_group):
        n = 0
        sum = 0
        for j in range(num_views):
            if group_scheme[i][j]:
                sum += scores[j]
                n += 1

        if n != 0:
            weight[i][0] = sum / n
            # weight[i][0] = tf.cast(tf.div(sum, n), dtype=tf.float32)

    return weight


def _view_pooling(final_view_descriptors, group_scheme):

    '''
    Final view descriptors are source of view pooling by grouping scheme.

    Use the average pooling (TODO: try max pooling later)

    :param group_scheme:
    :param final_view_descriptors:
    :return: group_descriptors
    '''

    group_descriptors = {}
    z_tensor = tf.zeros_like(final_view_descriptors[0])

    g_schemes = tf.unstack(group_scheme, axis=0)
    indices = [tf.squeeze(tf.where(elem), axis=1) for elem in g_schemes]
    for i, ind in enumerate(indices):
        view_desc = tf.cond(tf.not_equal(tf.size(ind), 0),
                            lambda : tf.gather(final_view_descriptors, ind),
                            lambda : tf.expand_dims(z_tensor, 0))
        group_descriptors[i] = tf.squeeze(tf.reduce_mean(view_desc, axis=0, keepdims=True), [0])

    return group_descriptors


def _weighted_fusion(group_descriptors, group_weight):
    '''
    To generate the shape level description, all these group
    level descriptors should be further combined.

    Get the final 3D shape descriptor D(S)

    :param
    group_descriptors: dic (average pooling per group - tensor)
    group_weight: tensor. shape=(5,1)

    :return:
    '''

    numerator = []  # 분자

    g_weight = tf.unstack(group_weight)
    n = len(g_weight)
    for i in range(n):
        numerator.append(tf.multiply(group_weight[i], group_descriptors[i]))

    denominator = tf.reduce_sum(group_weight)   # 분자

    return tf.div(tf.add_n(numerator), denominator)


def _CNN(inputs, is_training, dropout_keep_prob, reuse, scope, global_pool):
    '''
       The second part of the network (CNN) and the group module, are used to extract
       the final view descriptors together with the discrimination scores, separately.
    '''
    final_view_descriptors = []  # Final View Descriptors

    n_views = inputs.get_shape().as_list()[1]
    # transpose views: (NxVxHxWxC) -> (VxNxHxWxC)
    views = tf.transpose(inputs, perm=[1, 0, 2, 3, 4])

    with tf.variable_scope(scope, None, [inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            for i in range(n_views):
                batch_view = tf.gather(views, i)  # N x H x W x C

                net, end_points = \
                    inception_v2.inception_v2_base(batch_view, scope=scope)

                with tf.variable_scope('Logits'):
                    if global_pool:
                        # Global average pooling.
                        net = tf.reduce_mean(net, [1, 2], keepdims=True, name='global_pool')
                        end_points['global_pool'] = net
                    else:
                        # Pooling with a fixed kernel size.
                        kernel_size = inception_v2._reduced_kernel_size_for_small_input(net, [7, 7])
                        net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                              scope='AvgPool_1a_{}x{}'.format(*kernel_size))
                        end_points['AvgPool_1a'] = net

                    # ? x 1 x 1 x 1024
                    net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
                    # logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                    #                      normalizer_fn=None, scope='Conv2d_1c_1x1')
                    # if spatial_squeeze:
                    #     logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
                    end_points['Logits'] = net

                    final_view_descriptors.append(net)

    return final_view_descriptors


def _FCN(inputs, num_classes, reuse, scope):

    """
    Raw View Descriptor Generation

    first part of the network (FCN) to get the raw descriptor in the view level.
    The “FCN” part is the top five convolutional layers of GoogLeNet. (mid-level representation ??)

    Extract the raw view descriptors. Compared with deeper CNN,
    shallow FCN could have more position information, which is needed for
    the followed grouping module and the deeper CNN will have the content information
    which could represent the view feature better.

    Args:
    inputs: N x V x H x W x C tensor
    scope:
    """

    # FC layer to obtain the discrimination scores from raw view descriptors
    view_discrimination_scores = []

    n_views = inputs.get_shape().as_list()[1]

    # transpose views: (NxVxHxWxC) -> (VxNxHxWxC)
    views = tf.transpose(inputs, perm=[1, 0, 2, 3, 4])

    with tf.variable_scope(scope, None, [inputs], reuse=reuse) as scope:
        for i in range(n_views):
            batch_view = tf.gather(views, i)  # N x H x W x C
            # FCN
            logits = googLeNet.googLeNet(batch_view, num_classes, scope=scope)

            # The average score is shown for the batch size input
            # corresponding to each point of view.
            score = tf.nn.sigmoid(tf.log(tf.abs(logits)))
            score = tf.reduce_mean(tf.reduce_max(score, axis=1), axis=0)
            # score = tf.reshape(score, [])

            view_discrimination_scores.append(score)

    return view_discrimination_scores


def grouping_module(inputs,
                    num_classes,
                    reuse=tf.AUTO_REUSE,
                    scope='FCN'):

    view_discrimination_scores = _FCN(inputs,
                                      num_classes,
                                      reuse,
                                      scope)

    return view_discrimination_scores


def gvcnn(inputs,
          grouping_scheme,
          grouping_weight,
          num_classes,
          is_training=True,
          dropout_keep_prob=0.8,
          prediction_fn=slim.softmax,
          spatial_squeeze = True,
          reuse=tf.AUTO_REUSE,
          scope='GoogLeNet',
          global_pool=True):

    final_view_descriptors = _CNN(inputs,
                                  is_training,
                                  dropout_keep_prob,
                                  reuse,
                                  scope,
                                  global_pool)

    group_descriptors = _view_pooling(final_view_descriptors, grouping_scheme)
    shape_description = _weighted_fusion(group_descriptors, grouping_weight)

    # net = slim.flatten(shape_description)
    # logits = slim.fully_connected(net, num_classes, activation_fn=None)
    logits = slim.conv2d(shape_description, num_classes, [1, 1], activation_fn=None,
                         normalizer_fn=None, scope='Conv2d')
    if spatial_squeeze:
        logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')

    return logits


