from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np

from nets import inception_v4

slim = tf.contrib.slim

# The ore discriminative group should have higher weights and vice versa.
def grouping_scheme(view_discrimination_score, num_group, num_views):
    '''
    Note that 1 ≤ M ≤ N because there
    may exist sub-ranges that have no views falling into it.
    '''
    _grouping_scheme = np.full((num_group, num_views), False)

    for idx, s in enumerate(view_discrimination_score):
        if 0.0 <= s < 0.1:      # 0 group
            _grouping_scheme[0, idx] = True
        elif 0.1 <= s < 0.2:    # 1 group
            _grouping_scheme[1, idx] = True
        elif 0.2 <= s < 0.3:    # 2 group
            _grouping_scheme[2, idx] = True
        elif 0.3 <= s < 0.4:    # 3 group
            _grouping_scheme[3, idx] = True
        elif 0.4 <= s < 0.5:    # 4 group
            _grouping_scheme[4, idx] = True
        elif 0.5 <= s < 0.6:    # 5 group
            _grouping_scheme[5, idx] = True
        elif 0.6 <= s < 0.7:    # 6 group
            _grouping_scheme[6, idx] = True
        elif 0.7 <= s < 0.8:    # 7 group
            _grouping_scheme[7, idx] = True
        elif 0.8 <= s < 0.9:    # 8 group
            _grouping_scheme[8, idx] = True
        elif 0.9 <= s < 1.0:    # 9 group
            _grouping_scheme[9, idx] = True

    return _grouping_scheme


# TODO: modifiy func below
# currently, average value per group
def grouping_weight(view_discrimination_score, grouping_scheme):
    num_group = grouping_scheme.shape[0]
    num_views = grouping_scheme.shape[1]

    _grouping_weight = np.zeros(shape=(num_group, 1), dtype=np.float32)
    for i in range(num_group):
        n = 0
        sum = 0
        for j in range(num_views):
            if grouping_scheme[i][j]:
                sum += view_discrimination_score[j]
                n += 1

        if n != 0:
            _grouping_weight[i][0] = sum / n
            # weight[i][0] = tf.cast(tf.div(sum, n), dtype=tf.float32)

    return _grouping_weight


def view_pooling(final_view_descriptors, group_scheme):

    '''
    Intra-Group View Pooling

    Final view descriptors are source of view pooling with grouping scheme.

    Given the view descriptors and the generated grouping
    information, the objective here is to conduct intra-group
    view pooling towards a group level description.

    the views in the same group have the similar discrimination,
    which are assigned the same weight.

    Use max pooling

    :param group_scheme:
    :param final_view_descriptors:
    :return: group_descriptors
    '''
    group_descriptors = {}
    zero_tensor = tf.zeros_like(final_view_descriptors[0])

    group_scheme_lst = tf.unstack(group_scheme)
    indices = [tf.squeeze(tf.where(elem), axis=1) for elem in group_scheme_lst]
    for i, ind in enumerate(indices):
        view_desc = tf.cond(tf.not_equal(tf.size(ind), 0),
                            lambda : tf.gather(final_view_descriptors, ind),
                            lambda : tf.expand_dims(zero_tensor, 0))
        group_descriptors[i] = tf.squeeze(tf.reduce_max(view_desc, axis=0, keepdims=True), [0])

    return group_descriptors


def group_fusion(group_descriptors, group_weight):
    '''
    To generate the shape level description, all these group
    level descriptors should be further combined.

    The groups containing more discriminative views contribute more to
    the final 3D shape descriptor D(S) than those containing less discriminative views.
    By using these hierarchical view-group-shape description framework,
    the important and discriminative visual content can be discovered in the group level,
    and thus emphasized in the shape descriptor accordingly.

    :param
    group_descriptors: dic (average pooling per group - tensor)
    cls_group_weight: tensor. shape=(5,1)

    :return:
    '''
    group_weight_lst = tf.unstack(group_weight)
    numerator = []  # 분자
    for key, value in group_descriptors.items():
        numerator.append(tf.multiply(group_weight_lst[key], group_descriptors[key]))

    denominator = tf.reduce_sum(group_weight_lst)   # 분모
    shape_descriptor = tf.div(tf.add_n(numerator), denominator)

    return shape_descriptor


def discrimination_score(inputs,
                         num_classes,
                         is_training=True,
                         reuse=tf.AUTO_REUSE,
                         scope='InceptionV4'):
    """
    Raw View Descriptor Generation

    first part of the network (FCN) to get the raw descriptor in the view level.
    The “FCN” part is the top five convolutional layers of GoogLeNet.
    (mid-level representation)

    Extract the raw view descriptors.
    Compared with deeper CNN, shallow FCN could have more position information,
    which is needed for the followed grouping module and the deeper CNN will have
    the content information which could represent the view feature better.

    Args:
    inputs: N x V x H x W x C tensor
    scope:
    """
    raw_view_descriptors = []
    final_view_descriptors = []
    view_discrimination_scores = []

    n_views = inputs.get_shape().as_list()[1]
    # transpose views: (NxVxHxWxC) -> (VxNxHxWxC)
    views = tf.transpose(inputs, perm=[1, 0, 2, 3, 4])
    for index in range(n_views):
        batch_view = tf.gather(views, index)  # N x H x W x C
        with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
            raw_desc, net, end_points = \
                inception_v4.inception_v4(batch_view, v_scope='_view' + str(index),
                                          is_training=is_training, reuse=reuse, scope=scope)

        raw_view_descriptors.append(raw_desc['raw_desc'])
        final_view_descriptors.append(net)

        # GAP layer to obtain the discrimination scores from raw view descriptors.
        raw = tf.reduce_mean(raw_desc['raw_desc'], [1, 2], keep_dims=True)
        raw = slim.conv2d(raw, num_classes, [1, 1], activation_fn=None)
        raw = tf.reduce_max(raw, axis=[1,2,3])
        batch_view_score = tf.nn.sigmoid(tf.log(tf.abs(raw)))

        view_discrimination_scores.append(batch_view_score)

    # # Print name and shape of parameter nodes  (values not yet initialized)
    # tf.logging.info("++++++++++++++++++++++++++++++++++")
    # tf.logging.info("Parameters")
    # tf.logging.info("++++++++++++++++++++++++++++++++++")
    # for v in slim.get_model_variables():
    #     tf.logging.info('name = %s, shape = %s' % (v.name, v.get_shape()))

    return view_discrimination_scores, raw_view_descriptors, final_view_descriptors


def gvcnn(final_view_descriptors,
          grouping_scheme,
          grouping_weight,
          num_classes,
          is_training=True,
          dropout_keep_prob=0.8,
          create_aux_logits=False):

    # Intra-Group View Pooling
    group_descriptors = view_pooling(final_view_descriptors, grouping_scheme)
    # Group Fusion
    shape_descriptor = group_fusion(group_descriptors, grouping_weight)

    # (?, 8, 8, 1536)
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                        stride=1, padding='SAME'):
        # Final pooling and prediction
        with tf.variable_scope('Logits'):
            # 8 x 8 x 1536
            kernel_size = shape_descriptor.get_shape()[1:3]
            if kernel_size.is_fully_defined():
                net = slim.avg_pool2d(shape_descriptor, kernel_size, padding='VALID',
                                      scope='AvgPool_1a')
            else:
                net = tf.reduce_mean(shape_descriptor, [1, 2], keep_dims=True,
                                     name='global_pool')
            # end_points['global_pool'] = net
            if not num_classes:
                return net  #, end_points
            # 1 x 1 x 1536
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='Dropout_1b')
            net = slim.flatten(net, scope='PreLogitsFlatten')
            # end_points['PreLogitsFlatten'] = net
            # 1536
            logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                          scope='Logits')
            # end_points['Logits'] = logits  # (?,7)
            # end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')

    return logits, shape_descriptor
