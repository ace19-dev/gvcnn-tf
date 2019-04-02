from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np

from nets import inception_v4
from nets import resnet_model

slim = tf.contrib.slim


class MyModel(resnet_model.Model):
    def __init__(self, resnet_size, data_format='channels_last', num_classes=1000,
                 resnet_version=resnet_model.DEFAULT_VERSION,
                 dtype=resnet_model.DEFAULT_DTYPE):
        """These are the parameters that work for Imagenet data.

            Args:
              resnet_size: The number of convolutional layers needed in the model.
              data_format: Either 'channels_first' or 'channels_last', specifying which
                data format to use when setting up the model.
              num_classes: The number of output classes needed from the model. This
                enables users to extend the same model to their own datasets.
              resnet_version: Integer representing which version of the ResNet network
                to use. See README for details. Valid values: [1, 2]
              dtype: The TensorFlow dtype to use for calculations.
            """

        # For bigger models, we want to use "bottleneck" layers
        if resnet_size < 50:
            bottleneck = False
        else:
            bottleneck = True

        super(MyModel, self).__init__(
            resnet_size=resnet_size,
            bottleneck=bottleneck,
            num_classes=num_classes,
            num_filters=64,
            kernel_size=7,
            conv_stride=2,
            first_pool_size=3,
            first_pool_stride=2,
            block_sizes=_get_block_sizes(resnet_size),
            block_strides=[1, 2, 2, 2],
            resnet_version=resnet_version,
            data_format=data_format,
            dtype=dtype
        )


def _get_block_sizes(resnet_size):
    """Retrieve the size of each block_layer in the ResNet model.

    The number of block layers used for the Resnet model varies according
    to the size of the model. This helper grabs the layer set we want, throwing
    an error if a non-standard size has been selected.

    Args:
      resnet_size: The number of convolutional layers needed in the model.

    Returns:
      A list of block sizes to use in building the model.

    Raises:
      KeyError: if invalid resnet_size is received.
    """
    choices = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [3, 24, 36, 3]
    }

    try:
        return choices[resnet_size]
    except KeyError:
        err = ('Could not find layers for selected Resnet size.\n'
               'Size received: {}; sizes allowed: {}.'.format(
            resnet_size, choices.keys()))
        raise ValueError(err)


def grouping_scheme(view_discrimination_score, num_group, num_views):
    '''
    Note that 1 ≤ M ≤ N because there
    may exist sub-ranges that have no views falling into it.
    '''
    _grouping_scheme = np.full((num_group, num_views), False)

    for idx, s in enumerate(view_discrimination_score):
        if 0.0 <= s < 0.125:      # 0 group
            _grouping_scheme[0, idx] = True
        elif 0.125 <= s < 0.25:    # 1 group
            _grouping_scheme[1, idx] = True
        elif 0.25 <= s < 0.375:    # 2 group
            _grouping_scheme[2, idx] = True
        elif 0.375 <= s < 0.5:    # 3 group
            _grouping_scheme[3, idx] = True
        elif 0.5 <= s < 0.625:    # 4 group
            _grouping_scheme[4, idx] = True
        elif 0.625 <= s < 0.75:    # 5 group
            _grouping_scheme[5, idx] = True
        elif 0.75 <= s < 0.825:    # 6 group
            _grouping_scheme[6, idx] = True
        elif 0.825 <= s < 1.0:    # 7 group
            _grouping_scheme[7, idx] = True

    return _grouping_scheme


# TODO: will modifiy according to equation2 in paper correctly when totally understand.
# average value per group
def grouping_weight(view_discrimination_score, grouping_scheme):
    num_group = grouping_scheme.shape[0]    # 10
    num_views = grouping_scheme.shape[1]    # 8

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


def _view_pooling(final_view_descriptors, group_scheme):

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


def _group_fusion(group_descriptors, group_weight):
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
                         is_training=True,
                         reuse=tf.AUTO_REUSE,
                         scope='fcn'):
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
    # FC layer to obtain the discrimination scores from raw view descriptors
    view_discrimination_score = []
    raw_view_descriptors = []

    n_views = inputs.get_shape().as_list()[1]
    # transpose views: (NxVxHxWxC) -> (VxNxHxWxC)
    views = tf.transpose(inputs, perm=[1, 0, 2, 3, 4])

    with tf.variable_scope(scope, 'fcn', [inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            for i in range(n_views):
                batch_view = tf.gather(views, i)  # N x H x W x C
                # FCN
                raw_view, _ = inception_v4.fcn(batch_view, scope=scope)
                raw_view_descriptors.append(raw_view)

                # The average score is shown for the batch size input
                # corresponding to each point of view.
                batch_view_score = tf.nn.sigmoid(tf.log(tf.abs(raw_view)))
                batch_view_score = tf.reduce_mean(batch_view_score)
                # batch_view_score = tf.reshape(batch_view_score, [])

                view_discrimination_score.append(batch_view_score)

    return view_discrimination_score, raw_view_descriptors


def gvcnn(inputs,
          grouping_scheme,
          grouping_weight,
          num_classes,
          is_training=True,
          dropout_keep_prob=0.8,
          reuse=tf.AUTO_REUSE,
          scope='cnn',
          create_aux_logits=False):
    '''
    The second part of the network (CNN) and the group module, are used to extract
    the final view descriptors together with the discrimination scores, separately.
    '''
    with tf.variable_scope(scope, 'cnn', [inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):

            final_view_descriptors = []

            n_batchs = inputs.get_shape().as_list()[0]
            for i in range(n_batchs):
                batch_view = tf.gather(inputs, i)  # N x H x W x C

                net, end_points = inception_v4.cnn(batch_view, scope=scope)
                final_view_descriptors.append(net)

            # Intra-Group View Pooling
            group_descriptors = _view_pooling(final_view_descriptors, grouping_scheme)
            # Group Fusion
            shape_descriptor = _group_fusion(group_descriptors, grouping_weight)

            # (?, 8, 8, 1536)
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):
                # Auxiliary Head logits
                if create_aux_logits and num_classes:
                    with tf.variable_scope('AuxLogits'):
                        # 17 x 17 x 1024
                        aux_logits = end_points['Mixed_6h']
                        aux_logits = slim.avg_pool2d(aux_logits, [5, 5], stride=3,
                                                     padding='VALID',
                                                     scope='AvgPool_1a_5x5')
                        aux_logits = slim.conv2d(aux_logits, 128, [1, 1],
                                                 scope='Conv2d_1b_1x1')
                        aux_logits = slim.conv2d(aux_logits, 768,
                                                 aux_logits.get_shape()[1:3],
                                                 padding='VALID', scope='Conv2d_2a')
                        aux_logits = slim.flatten(aux_logits)
                        aux_logits = slim.fully_connected(aux_logits, num_classes,
                                                          activation_fn=None,
                                                          scope='Aux_logits')
                        end_points['AuxLogits'] = aux_logits   # (?,7)

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
                    end_points['global_pool'] = net
                    if not num_classes:
                        return net, end_points
                    # 1 x 1 x 1536
                    net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b')
                    net = slim.flatten(net, scope='PreLogitsFlatten')
                    end_points['PreLogitsFlatten'] = net
                    # 1536
                    logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                                  scope='Logits')
                    end_points['Logits'] = logits   # (?,7)
                    end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')

        return logits, shape_descriptor, end_points



