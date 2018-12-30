import time
import numpy as np
import cv2
import math

import tensorflow as tf
import gvcnn
import _model
import data

from utils import train_utils
from nets import googLeNet

slim = tf.contrib.slim

# prefetch_queue = slim.prefetch_queue

flags = tf.app.flags

FLAGS = flags.FLAGS

# LABELS_CLASS = 'labels_class'
IMAGE = 'image'
# HEIGHT = 'height'
# WIDTH = 'width'
# IMAGE_NAME = 'image_name'
LABEL = 'label'
OUTPUT_TYPE = 'classification'

flags.DEFINE_integer('num_clones', 1, 'Number of clones to deploy.')
flags.DEFINE_boolean('clone_on_cpu', False, 'Use CPUs to deploy clones.')
flags.DEFINE_integer('num_replicas', 1, 'Number of worker replicas.')
flags.DEFINE_integer('startup_delay_steps', 15,
                     'Number of training steps between replicas startup.')
flags.DEFINE_integer('num_ps_tasks', 0,
                     'The number of parameter servers. If the value is 0, '
                     'then the parameters are handled locally by the worker.')
flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')
flags.DEFINE_integer('task', 0, 'The task ID.')

# Settings for logging.
flags.DEFINE_string('train_logdir', './checkpoint',
                    'Where the checkpoint and logs are stored.')
flags.DEFINE_integer('log_steps', 10,
                     'Display logging information at every log_steps.')
flags.DEFINE_integer('save_interval_secs', 1200,
                     'How often, in seconds, we save the model to disk.')
flags.DEFINE_boolean('save_summaries_images', True,
                     'Save sample inputs, labels, and semantic predictions as '
                     'images to summary.')

flags.DEFINE_enum('learning_policy', 'poly', ['step'],
                  'Learning rate policy for training.')
flags.DEFINE_float('base_learning_rate', .0001,
                   'The base learning rate for model training.')
flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                   'The rate to decay the base learning rate.')
flags.DEFINE_float('learning_rate_decay_step', .2000,
                   'Decay the base learning rate at a fixed step.')
flags.DEFINE_float('learning_power', 0.9,
                   'The power value used in the poly learning policy.')
flags.DEFINE_float('training_number_of_steps', 30000,
                   'The number of steps used for training.')
flags.DEFINE_float('momentum', 0.9, 'The momentum value to use')

# flags.DEFINE_integer('train_batch_size', 8,
#                      'The value of the weight decay for training.')


# Settings for fine-tuning the network.
flags.DEFINE_string('tf_initial_checkpoint', './checkpoint',
                    'The initial checkpoint in tensorflow format.')

# Dataset settings.
flags.DEFINE_string('dataset_dir', '/home/ace19/dl_data/modelnet',
                    'Where the dataset reside.')

flags.DEFINE_integer('how_many_training_epochs', 3, 'How many training loops to run')

flags.DEFINE_integer('batch_size', 16, 'batch size')
flags.DEFINE_integer('num_views', 8, 'number of views')
flags.DEFINE_string('height_weight', '224,224', 'height and weight')
flags.DEFINE_integer('num_classes', 7, 'number of classes')
flags.DEFINE_integer('num_group', 5, 'number of grouping')


def test():
    h_w = list(map(int, FLAGS.height_weight.split(',')))

    x = tf.placeholder(tf.float32, [None, FLAGS.num_views, h_w[0], h_w[1], 3])
    ground_truth = tf.placeholder(tf.int32, [None])
    is_training = tf.placeholder(tf.bool)
    dropout_keep_prob = tf.placeholder(tf.float32)
    group_scheme = tf.placeholder(tf.bool, [FLAGS.num_group, FLAGS.num_views])
    group_weight = tf.placeholder(tf.float32, [FLAGS.num_group, 1])

    # Making grouping module
    d_scores, g_scheme = \
        gvcnn.make_grouping_module(x,
                                   FLAGS.num_group,
                                   is_training,
                                   dropout_keep_prob=dropout_keep_prob)

    # GVCNN
    predictions = gvcnn.gvcnn(x,
                              group_scheme,
                              group_weight,
                              FLAGS.num_classes,
                              is_training,
                              dropout_keep_prob=dropout_keep_prob)

    with tf.Session() as sess:
        # temporary data for test
        inputs = tf.random_uniform((FLAGS.train_batch_size, FLAGS.num_views, h_w[0], h_w[1], 3))
        scores, scheme = \
            sess.run([d_scores, g_scheme], feed_dict={x: inputs.eval(),
                                                      is_training: True,
                                                      dropout_keep_prob: 0.8})

        g_scheme = gvcnn.refine_group(scheme, FLAGS.num_group, FLAGS.num_views)
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


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    # test()
    # test2()
    # test3()
    # test4()

    h_w = list(map(int, FLAGS.height_weight.split(',')))

    tf.gfile.MakeDirs(FLAGS.train_logdir)

    with tf.Graph().as_default() as graph:
        dataset = data.Data(FLAGS.dataset_dir, h_w)

        global_step = tf.train.get_or_create_global_step()

        x = tf.placeholder(tf.float32, [None, FLAGS.num_views, h_w[0], h_w[1], 3])
        ground_truth = tf.placeholder(tf.int32, [None])
        is_training = tf.placeholder(tf.bool)
        dropout_keep_prob = tf.placeholder(tf.float32)
        group_scheme = tf.placeholder(tf.bool, [FLAGS.num_group, FLAGS.num_views])
        group_weight = tf.placeholder(tf.float32, [FLAGS.num_group, 1])

        # Making grouping module
        d_scores, g_scheme = \
            gvcnn.make_grouping_module(x,
                                       FLAGS.num_group,
                                       is_training,
                                       dropout_keep_prob=dropout_keep_prob)

        # GVCNN
        predictions = gvcnn.gvcnn(x,
                                  group_scheme,
                                  group_weight,
                                  FLAGS.num_classes,
                                  is_training,
                                  dropout_keep_prob=dropout_keep_prob)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # Add summaries for model variables.
        for model_var in slim.get_model_variables():
            summaries.add(tf.summary.histogram(model_var.op.name, model_var))

        # Add summaries for images, labels, semantic predictions
        if FLAGS.save_summaries_images:
            summary_image = graph.get_tensor_by_name(
                ('%s:0' % (IMAGE)).strip('/'))
            summaries.add(
                tf.summary.image('samples/%s' % IMAGE, summary_image))

            label = graph.get_tensor_by_name(
                # ('%s/%s:0' % (scope, LABEL)).strip('/'))
                ('%s:0' % (LABEL)).strip('/'))
            # Scale up summary image pixel values for better visualization.
            pixel_scaling = max(1, 255 // dataset.num_classes)
            summary_label = tf.cast(label * pixel_scaling, tf.uint8)
            summaries.add(
                tf.summary.image('samples/%s' % LABEL, summary_label))

            output = graph.get_tensor_by_name(
                # ('%s/%s:0' % (scope, OUTPUT_TYPE)).strip('/'))
                ('%s:0' % (OUTPUT_TYPE)).strip('/'))
            predictions = tf.expand_dims(tf.argmax(output, 3), -1)

            summary_predictions = tf.cast(predictions * pixel_scaling, tf.uint8)
            summaries.add(
                tf.summary.image(
                    'samples/%s' % OUTPUT_TYPE, summary_predictions))

        # Add summaries for losses.
        for loss in tf.get_collection(tf.GraphKeys.LOSSES):
            summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

        # Build the optimizer
        learning_rate = train_utils.get_model_learning_rate(
            FLAGS.learning_policy, FLAGS.base_learning_rate,
            FLAGS.learning_rate_decay_step, FLAGS.learning_rate_decay_factor,
            None, FLAGS.learning_power,
            FLAGS.slow_start_step, FLAGS.slow_start_learning_rate)
        optimizer = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)
        summaries.add(tf.summary.scalar('learning_rate', learning_rate))

        for variable in slim.get_model_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))

        total_loss, grads_and_vars = train_utils.optimize(optimizer)
        total_loss = tf.check_numerics(total_loss, 'Loss is inf or nan.')
        summaries.add(tf.summary.scalar('total_loss', total_loss))

        # Modify the gradients for biases and last layer variables.
        last_layers = train_utils.get_extra_layer_scopes(
            FLAGS.last_layers_contain_logits_only)
        grad_mult = train_utils.get_model_gradient_multipliers(
            last_layers, FLAGS.last_layer_gradient_multiplier)
        if grad_mult:
            grads_and_vars = slim.learning.multiply_gradients(
                grads_and_vars, grad_mult)

        # Create gradient update op.
        grad_updates = optimizer.apply_gradients(
            grads_and_vars, global_step=global_step)
        update_ops.append(grad_updates)
        update_op = tf.group(*update_ops)
        with tf.control_dependencies([update_op]):
            train_tensor = tf.identity(total_loss, name='train_op')

        # Add the summaries from the first clone. These contain the summaries
        # created by model_fn and either optimize_clones() or _gather_clone_loss().
        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries))
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', graph)

        # Create a saver object which will save all the variables
        saver = tf.train.Saver()

        '''
        TODO: accuracy, loss, read/save checkpoint  ...
        '''


        start_epoch = 0
        # Get the number of training/validation steps per epoch
        t_batches = int(dataset.data_size() / FLAGS.batch_size)
        if dataset.data_size() % FLAGS.batch_size > 0:
            t_batches += 1
        # v_batches = int(dataset.data_size() / FLAGS.batch_size)
        # if val_data.data_size() % FLAGS.batch_size > 0:
        #     v_batches += 1

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        ############################
        # Training loop.
        ############################
        for training_epoch in range(start_epoch, FLAGS.how_many_training_epochs):
            print("------------------------")
            print(" Epoch:{} >> ".format(training_epoch + 1))
            print("------------------------")

            dataset.shuffle()
            for step in range(t_batches):
                # Pull the image batch we'll use for training.
                train_batch_xs, train_batch_ys = dataset.next_batch(step, FLAGS.batch_size)

                # For debugging
                # img = train_batch_xs[0]
                # plt.hist(img.ravel())
                # plt.show()

                # Run the graph with this batch of training data.
                # train_summary, train_accuracy, cross_entropy_value, _, _ = \
                #     sess.run(
                #         [
                #             merged_summaries, accuracy, cross_entropy_mean, train_step,
                #             increment_global_step
                #         ],
                #         feed_dict={
                #             X: train_batch_xs,
                #             ground_truth: train_batch_ys,
                #             learning_rate_input: learning_rate_value,
                #             phase_train: True,
                #             # momentum: 0.95,
                #             # dropout_prob: 0.5
                #         }
                #     )
                # ====================================================================
                scores, scheme = \
                    sess.run([d_scores, g_scheme], feed_dict={x: train_batch_xs,
                                                              is_training: True,
                                                              dropout_keep_prob: 0.8})

                g_scheme = gvcnn.refine_group(scheme, FLAGS.num_group, FLAGS.num_views)
                g_weight = gvcnn.group_weight(scores, g_scheme)
                pred = sess.run([predictions], feed_dict={x: train_batch_xs,
                                                          ground_truth: train_batch_ys,
                                                          group_scheme: g_scheme,
                                                          group_weight: g_weight,
                                                          is_training: True,
                                                          dropout_keep_prob: 0.8})

                tf.logging.info("pred...%s", pred)

                # train_writer.add_summary(train_summary, training_epoch)
                # tf.logging.info('Epoch #%d, Step #%d, rate %f, accuracy %.1f%%, cross entropy %f' %
                #                 (training_epoch, step, learning_rate_value, train_accuracy * 100,
                #                  cross_entropy_value))

            ##############################################
            # Validate the model on the validation set
            ##############################################


if __name__ == '__main__':
    tf.app.run()
