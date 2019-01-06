from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np

from slim.nets import inception_v2
from nets import googLeNet

slim = tf.contrib.slim


# TODO: The more grouping, the better result ??
def grouping_scheme(view_discrimination_scores, num_group, num_views):
    grouping_schemes = []

    for _, scores in enumerate(view_discrimination_scores):
        g = np.full((num_group, num_views), False)

        for idx, s in enumerate(scores):
            if 0.0 <= s < 0.1:      # 0 group
                g[0, idx] = True
            elif 0.1 <= s < 0.2:    # 1 group
                g[1, idx] = True
            elif 0.2 <= s < 0.3:    # 2 group
                g[2, idx] = True
            elif 0.3 <= s < 0.4:    # 3 group
                g[3, idx] = True
            elif 0.4 <= s < 0.5:    # 4 group
                g[4, idx] = True
            elif 0.5 <= s < 0.6:    # 5 group
                g[5, idx] = True
            elif 0.6 <= s < 0.7:    # 6 group
                g[6, idx] = True
            elif 0.7 <= s < 0.8:    # 7 group
                g[7, idx] = True
            elif 0.8 <= s < 0.9:    # 8 group
                g[8, idx] = True
            elif 0.9 <= s < 1.0:    # 9 group
                g[9, idx] = True

        grouping_schemes.append(g)

    return grouping_schemes


# TODO: modified according to equation2 in paper correctly when totally understand.
# average value per group
def grouping_weight(view_discrimination_scores, grouping_schemes):

    grouping_weights = []
    for idx, grouping_scheme in enumerate(grouping_schemes):
        num_group = grouping_scheme.shape[0]
        num_views = grouping_scheme.shape[1]

        weight = np.zeros((num_group, 1), dtype=np.float32)
        for i in range(num_group):
            n = 0
            sum = 0
            for j in range(num_views):
                if grouping_scheme[i][j]:
                    sum += view_discrimination_scores[idx][j]
                    n += 1

            if n != 0:
                weight[i][0] = sum / n
                # weight[i][0] = tf.cast(tf.div(sum, n), dtype=tf.float32)

        grouping_weights.append(weight)

    return grouping_weights


def _view_pooling(final_view_descriptors, group_scheme):

    '''
    Final view descriptors are source of view pooling with grouping scheme.

    Use the average pooling (TODO: try max pooling later)

    :param group_scheme:
    :param final_view_descriptors:
    :return: group_descriptors
    '''

    cls_group_descriptors = []
    # for cls_view_descriptors in final_view_descriptors:
    group_descriptors = {}
    zero_tensor = tf.zeros_like(final_view_descriptors[0][0])

    cls_g_schemes = tf.unstack(group_scheme, axis=0)
    for idx, g_schemes in enumerate(cls_g_schemes):
        x = tf.unstack(g_schemes)
        indices = [tf.squeeze(tf.where(elem), axis=1) for elem in x]
        for i, ind in enumerate(indices):
            view_desc = tf.cond(tf.not_equal(tf.size(ind), 0),
                                lambda : tf.gather(final_view_descriptors[idx], ind),
                                lambda : tf.expand_dims(zero_tensor, 0))
            group_descriptors[i] = tf.squeeze(tf.reduce_mean(view_desc, axis=0, keepdims=True), [0])

        cls_group_descriptors.append(group_descriptors)

    return cls_group_descriptors


def _group_fusion(cls_group_descriptors, cls_group_weight):
    '''
    To generate the shape level description, all these group
    level descriptors should be further combined.

    Get the final 3D shape descriptor D(S)

    :param
    group_descriptors: dic (average pooling per group - tensor)
    cls_group_weight: tensor. shape=(5,1)

    :return:
    '''
    cls_shape_descriptor = []

    cls_group_weight_lst = tf.unstack(cls_group_weight)
    for idx, group_descriptors in enumerate(cls_group_descriptors):
        group_weight_lst = tf.unstack(cls_group_weight_lst[idx])
        numerator = []  # 분자
        for key, value in group_descriptors.items():
            numerator.append(tf.multiply(group_weight_lst[key], group_descriptors[key]))

        denominator = tf.reduce_sum(group_weight_lst)   # 분모
        shape_descriptor = tf.div(tf.add_n(numerator), denominator)
        cls_shape_descriptor.append(shape_descriptor)

    return tf.concat(cls_shape_descriptor, axis=0)


def _CNN(inputs, is_training, dropout_keep_prob, reuse, scope, global_pool):
    '''
       The second part of the network (CNN) and the group module, are used to extract
       the final view descriptors together with the discrimination scores, separately.
    '''
    final_view_descriptors = []  # Final View Descriptors on cls

    cls_batch_input = tf.unstack(inputs, axis=0)
    for inputs in cls_batch_input:

        n_views = inputs.get_shape().as_list()[1]
        # transpose views: (NxVxHxWxC) -> (VxNxHxWxC)
        views = tf.transpose(inputs, perm=[1, 0, 2, 3, 4])

        view_descriptors = []
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
                        # net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
                        # ? x 1 x 1 x 1024
                        # logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                        #                      normalizer_fn=None, scope='Conv2d_1c_1x1')
                        # if spatial_squeeze:
                        #     logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
                        end_points['Logits'] = net

                    view_descriptors.append(net)

        final_view_descriptors.append(view_descriptors)

    return final_view_descriptors


def discrimination_score(inputs, num_classes, reuse=tf.AUTO_REUSE, scope='FCN'):

    """
    Raw View Descriptor Generation

    first part of the network (FCN) to get the raw descriptor in the view level.
    The “FCN” part is the top five convolutional layers of GoogLeNet. (mid-level representation ??)

    Extract the raw view descriptors. Compared with deeper CNN,
    shallow FCN could have more position information, which is needed for
    the followed grouping module and the deeper CNN will have the content information
    which could represent the view feature better.

    Args:
    inputs: C X N x V x H x W x C tensor
    scope:
    """

    # FC layer to obtain the discrimination scores from raw view descriptors
    view_discrimination_scores = []

    cls_batch_input = tf.unstack(inputs, axis=0)
    for batch_inputs in cls_batch_input:

        n_views = batch_inputs.get_shape().as_list()[1]
        # transpose views: (NxVxHxWxC) -> (VxNxHxWxC)
        views = tf.transpose(batch_inputs, perm=[1, 0, 2, 3, 4])

        scores = []
        with tf.variable_scope(scope, None, [batch_inputs], reuse=reuse) as scope:
            for i in range(n_views):
                batch_view = tf.gather(views, i)  # N x H x W x C
                # FCN
                logits = googLeNet.googLeNet(batch_view, num_classes, scope=scope)

                # The average score is shown for the batch size input
                # corresponding to each point of view.
                batch_view_score = tf.nn.sigmoid(tf.log(tf.abs(logits)))
                batch_view_score = tf.reduce_mean(tf.reduce_max(batch_view_score, axis=1), axis=0)
                # batch_view_score = tf.reshape(batch_view_score, [])

                scores.append(batch_view_score)

        view_discrimination_scores.append(scores)

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
    shape_descriptor = _group_fusion(group_descriptors, grouping_weight)

    # net = slim.flatten(shape_description)
    # logits = slim.fully_connected(net, num_classes, activation_fn=None)
    logits = slim.conv2d(shape_descriptor, num_classes, [1, 1], activation_fn=None,
                         normalizer_fn=None, scope=scope)
    if spatial_squeeze:
        logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')

    return logits


