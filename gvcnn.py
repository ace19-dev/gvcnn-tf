from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np

from slim.nets import inception_v2
from nets import googLeNet

slim = tf.contrib.slim


# NUM_GROUP = 5


def _scheme_group(group_scheme):
    ret_val = np.full((5, 8), False)

    for key, value in group_scheme.items():
        for v in group_scheme[key]:
            ret_val[key-1,v] = True

    return ret_val


def refine_group(group_scheme):
    new_scheme = {}
    for i, g in enumerate(group_scheme):
        try:
            new_scheme[g].append(i)
        except KeyError:
            new_scheme[g] = [i]

    return _scheme_group(new_scheme)


# TODO: weight will be modified according to equation2 in paper correctly
# average value per group
def group_weight(scores, group_scheme):
    num_group = group_scheme.shape[0]
    num_views = group_scheme.shape[1]

    weight = np.zeros((num_group, 1), dtype=np.float32)

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

    Use the average pooling (TODO: check max pooling later)

    :param group_scheme:
    :param final_view_descriptors:
    :return:
    '''

    group_descriptors = {}

    g_schemes = tf.unstack(group_scheme)
    indices_list = [tf.squeeze(tf.where(elem)) for elem in g_schemes]
    for i, indices in enumerate(indices_list):
        view_desc = tf.gather(final_view_descriptors, indices)
        # tf.reduce_max()
        group_descriptors[i] = tf.reduce_mean(view_desc, 0)

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
    '''
    '''
    numerator = []  # 분자
    denominator = []  # 분모
    # for key, value in group_descriptors.items():

    g_weight = tf.unstack(group_weight)
    for i in range(len(g_weight)):
        numerator.append(tf.multiply(group_weight[i], group_descriptors[i]))

    denominator = tf.reduce_sum(group_weight)

    return tf.div(tf.add_n(numerator), denominator)


# TODO: modify func dynamically
def grouping_scheme(num_group, view_discrimination_scores):
    group = []

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

        group.append(group_idx)

    return group

# TODO: modify FCN soon
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

    The “FCN” part is the top five convolutional layers of GoogLeNet. (mid-level representation ??)

    Extract the raw view descriptors. Compared with deeper CNN,
    shallow FCN could have more position information, which is needed for
    the followed grouping module and the deeper CNN will have the content information
    which could represent the view feature better.

    Args:
    inputs: N x V x H x W x C tensor
    scope:
    """

    input_views = []
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

                    # ? X 1 x 1 x 1024
                    net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b')
                    net = slim.flatten(net)
                    # net = slim.fully_connected(net, 512)
                    logits = slim.fully_connected(net, 1, activation_fn=None)
                    end_points['Logits'] = logits

                    # The average score is shown for the batch size input
                    # corresponding to each point of view.
                    score = tf.nn.sigmoid(tf.log(tf.abs(logits)))
                    score = tf.reduce_mean(score, axis=0)
                    score = tf.reshape(score, [])

                    view_discrimination_scores.append(score)

    g_scheme = grouping_scheme(num_group, view_discrimination_scores)

    return view_discrimination_scores, g_scheme


def gvcnn(inputs,
          group_scheme,
          group_weight,
          num_classes,
          is_training=True,
          dropout_keep_prob=0.8,
          prediction_fn=slim.softmax,
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

                    # ? x 1 x 1 x 1024
                    net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
                    # logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                    #                      normalizer_fn=None, scope='Conv2d_1c_1x1')
                    # if spatial_squeeze:
                    #     logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
                    end_points['Logits'] = net

                    final_view_descriptors.append(net)

    group_descriptors = _view_pooling(final_view_descriptors, group_scheme)
    shape_description = _weighted_fusion(group_descriptors, group_weight)

    net = slim.flatten(shape_description)
    logits = slim.fully_connected(net, num_classes, activation_fn=None)
    # logits = slim.conv2d(shape_description, num_classes, [1, 1], activation_fn=None,
    #                      normalizer_fn=None, scope='Conv2d')
    # logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')

    pred = prediction_fn(logits, scope='Predictions')

    return pred


