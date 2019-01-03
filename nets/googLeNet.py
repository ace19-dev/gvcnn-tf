"""
A simple implementation of GoogLeNet
"""

import numpy as np
import tensorflow as tf

slim = tf.contrib.slim


# network structure
def _inception(x, p1, p2, p3, p4, scope):
    p1f11 = p1
    p2f11, p2f33 = p2
    p3f11, p3f55 = p3
    p4f11 = p4
    with tf.variable_scope(scope):
        path1 = tf.layers.conv2d(x, filters=p1f11, kernel_size=1, activation=tf.nn.relu, name='p1f11')

        path2 = tf.layers.conv2d(x, p2f11, 1, activation=tf.nn.relu, name='p2f11')
        path2 = tf.layers.conv2d(path2, p2f33, 3, padding='same', activation=tf.nn.relu, name='p2f33')

        path3 = tf.layers.conv2d(x, p3f11, 1, activation=tf.nn.relu, name='p3f11')
        path3 = tf.layers.conv2d(path3, p3f55, 5, padding='same', activation=tf.nn.relu, name='p3f55')

        path4 = tf.layers.max_pooling2d(x, pool_size=3, strides=1, padding='same', name='p4p33')
        path4 = tf.layers.conv2d(path4, p4f11, 1, activation=tf.nn.relu, name='p4f11')

        out = tf.concat((path1, path2, path3, path4), axis=-1, name='path_cat')
    return out


'''
    (batch, height, width, channel)
'''
def googLeNet(image, num_classes, scope='GoogLeNet'):
    with tf.variable_scope(scope):
        net = tf.layers.conv2d(
            inputs=image,
            filters=12,
            kernel_size=5,
            strides=1,
            padding='same',
            name="conv1")
        net = tf.layers.max_pooling2d(net, 2, 2, name="maxpool1")
        net = _inception(net, p1=64, p2=(6, 64), p3=(6, 32), p4=32, scope='incpt1')
        net = tf.layers.max_pooling2d(net, 3, 2, padding='same', name="maxpool1")
        net = _inception(net, p1=256, p2=(32, 256), p3=(32, 128), p4=128, scope='incpt2')
        net = tf.layers.average_pooling2d(net, 7, 1, name="avgpool")
        net = tf.layers.flatten(net, name='flat')
        logits = tf.layers.dense(net, num_classes, name='fc4')
        # TODO
        # logits = slim.conv2d(net, num_classes, [3, 3], stride=1, activation_fn=None,
        #                      normalizer_fn=None, scope='Conv2d_1c_1x1')

        return logits
