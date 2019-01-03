from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np

from slim.nets import inception_v2
from nets import googLeNet

slim = tf.contrib.slim


# NUM_GROUP = 5


def _mask_group_scheme(group_scheme, num_group, num_views):
    g = np.full((num_group, num_views), False)

    for key, _ in group_scheme.items():
        for view in group_scheme[key]:
            g[key-1, view] = True

    return g


def _search(lst, target):
  min = 0
  max = len(lst)-1
  avg = (min+max)//2
  # uncomment next line for traces
  # print lst, target, avg
  while (min < max):
    if (lst[avg] == target):
      return avg
    elif (lst[avg] < target):
      return avg + 1 + _search(lst[avg+1:], target)
    else:
      return _search(lst[:avg], target)

  # avg may be a partial offset so no need to print it here
  # print "The location of the number in the array is", avg
  return len(lst) - avg


def refine_group(scores, num_group, num_views):
    max_range = []
    for i in range(num_group):
        max_range.append(i / num_group)
    max_range.append(1.0)

    new_scheme = {}
    for idx, s in enumerate(scores):
        _search(max_range, s)
        # if s >= 0 and s < 2.5:
        #     try:
        #         new_scheme[0].append(idx)
        #     except KeyError:
        #         new_scheme[0] = [idx]
        # elif s >= 2.5 and s < 5:
        #     try:
        #         new_scheme[1].append(idx)
        #     except KeyError:
        #         new_scheme[1] = [idx]
        # elif s >= 5 and s < 6.5:
        #     try:
        #         new_scheme[2].append(idx)
        #     except KeyError:
        #         new_scheme[2] = [idx]
        # elif s >= 0.7 and s < 0.8:
        #     try:
        #         new_scheme[3].append(idx)
        #     except KeyError:
        #         new_scheme[3] = [idx]
        # elif s > 0.8 and s <= 0.9:
        #     try:
        #         new_scheme[4].append(idx)
        #     except KeyError:
        #         new_scheme[4] = [idx]
        # elif s > 0.9 and s <= 1:
        #     try:
        #         new_scheme[5].append(idx)
        #     except KeyError:
        #         new_scheme[5] = [idx]




    new_scheme = {}
    for i, g in enumerate(scores):
        try:
            new_scheme[g].append(i)
        except KeyError:
            new_scheme[g] = [i]

    return _mask_group_scheme(new_scheme, num_group, num_views)


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


# TODO: modify FCN
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
            # TODO: reduce_max ??
            score = tf.reduce_mean(score)
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

    # TODO: modify dynamically
    # group = []
    # g0 = tf.constant(0, dtype=tf.float32)
    # g1 = tf.constant(1 / num_group, dtype=tf.float32)
    # g2 = tf.constant(2 / num_group, dtype=tf.float32)
    # g3 = tf.constant(3 / num_group, dtype=tf.float32)
    # g4 = tf.constant(4 / num_group, dtype=tf.float32)
    # g5 = tf.constant(5 / num_group, dtype=tf.float32)
    #
    # for view_idx, score in enumerate(view_discrimination_scores):
    #     group_idx = tf.case(
    #         pred_fn_pairs=[
    #             (tf.logical_and(tf.greater_equal(score, g0), tf.less(score, g1)), lambda: tf.constant(1)),
    #             (tf.logical_and(tf.greater_equal(score, g1), tf.less(score, g2)), lambda: tf.constant(2)),
    #             (tf.logical_and(tf.greater_equal(score, g2), tf.less(score, g3)), lambda: tf.constant(3)),
    #             (tf.logical_and(tf.greater_equal(score, g3), tf.less(score, g4)), lambda: tf.constant(4)),
    #             (tf.logical_and(tf.greater_equal(score, g4), tf.less(score, g5)), lambda: tf.constant(5))],
    #         default=lambda: tf.constant(-1),
    #         exclusive=False)
    #     group.append(group_idx)
    #
    # return view_discrimination_scores, group


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


