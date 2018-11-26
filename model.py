from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from slim.nets import inception_v2

slim = tf.contrib.slim


def FCN(inputs,
        num_classes=1000,
        is_training=True,
        dropout_keep_prob=0.8,
        scope='GVCNN'):

    return inception_v2.inception_v2_base(inputs,
                                          final_endpoint='Mixed_5c',
                                          min_depth=16,
                                          depth_multiplier=1.0,
                                          scope=None)





def FC():





def raw_view_descriptors():





def view_pooling():





def group_fusion():
    return


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