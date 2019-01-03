import tensorflow as tf
import numpy as np

import gvcnn
from _ref import _model
from nets import googLeNet


def test(h_w, n_views, n_group, n_classes, n_batch):
    x = tf.placeholder(tf.float32, [None, n_views, h_w[0], h_w[1], 3])
    ground_truth = tf.placeholder(tf.int32, [None])
    is_training = tf.placeholder(tf.bool)
    dropout_keep_prob = tf.placeholder(tf.float32)
    group_scheme = tf.placeholder(tf.bool, [n_group, n_views])
    group_weight = tf.placeholder(tf.float32, [n_group, 1])

    # Making grouping module
    d_scores, g_scheme = \
        gvcnn.grouping_module(x,
                              n_group,
                              is_training,
                              dropout_keep_prob=dropout_keep_prob)

    # GVCNN
    predictions = gvcnn.gvcnn(x,
                              group_scheme,
                              group_weight,
                              n_classes,
                              is_training,
                              dropout_keep_prob=dropout_keep_prob)

    with tf.Session() as sess:
        # temporary data for test
        inputs = tf.random_uniform((n_batch, n_views, h_w[0], h_w[1], 3))
        scores, scheme = \
            sess.run([d_scores, g_scheme], feed_dict={x: inputs.eval(),
                                                      is_training: True,
                                                      dropout_keep_prob: 0.8})

        g_scheme = gvcnn.refine_group(scheme, n_group, n_views)
        g_weight = gvcnn.group_weight(scores, g_scheme)
        pred = sess.run([predictions], feed_dict={x: inputs.eval(),
                                                  group_scheme: g_scheme,
                                                  group_weight: g_weight,
                                                  is_training: True,
                                                  dropout_keep_prob: 0.8})

        tf.logging.info("pred...%s", pred)


def test2():
    train_batch_size = 8
    height, width = 224, 224

    inputs = tf.random_uniform((train_batch_size, height, width, 3))

    googLeNet.googLeNet(inputs)


def test3():
    train_batch_size = 1
    num_views = 8
    height, width = 224, 224

    inputs = tf.random_uniform((train_batch_size, num_views, height, width, 3))

    _model.inference_multiview(inputs, 10, 0.8)


def test4():
    group_descriptors = {}
    final_view_descriptors = []
    for i in range(5):
        input = tf.random_uniform((8, 1, 1, 1024))
        final_view_descriptors.append(input)

    empty = tf.zeros_like(final_view_descriptors[0])

    b = tf.constant([[True, False, True, False],
                     [False, False, False, True],
                     [False, False, False, False],
                     [False, True, False, False],
                     [False, False, False, False]])
    x = tf.unstack(b)
    indices = [tf.squeeze(tf.where(e), axis=1) for e in x]
    for i, ind in enumerate(indices):
        view_desc = tf.cond(tf.not_equal(tf.size(ind), 0),
                            lambda: tf.gather(final_view_descriptors, ind),
                            lambda: tf.expand_dims(empty, 0))
        group_descriptors[i] = tf.squeeze(tf.reduce_mean(view_desc, axis=0, keepdims=True), [0])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result, result2 = sess.run([indices, group_descriptors])
        print(result)
        print("...")
        print(result2)

