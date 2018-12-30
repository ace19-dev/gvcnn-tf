import time
import numpy as np
import cv2
import math


import tensorflow as tf
import gvcnn
import _model
import data

from nets import googLeNet


slim = tf.contrib.slim

# prefetch_queue = slim.prefetch_queue

flags = tf.app.flags

FLAGS = flags.FLAGS


# LABELS_CLASS = 'labels_class'
# IMAGE = 'image'
# HEIGHT = 'height'
# WIDTH = 'width'
# IMAGE_NAME = 'image_name'
# LABEL = 'label'


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


flags.DEFINE_float('base_learning_rate', .0001,
                   'The base learning rate for model training.')
flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                   'The rate to decay the base learning rate.')
flags.DEFINE_float('learning_rate_decay_step', .2000,
                   'Decay the base learning rate at a fixed step.')
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


    '''
    TODO: accuracy, loss, summary, read/save checkpoint  ...
    '''


    dataset = data.Data(FLAGS.dataset_dir, h_w)

    start_epoch = 0
    # Get the number of training/validation steps per epoch
    t_batches = int(dataset.data_size() / FLAGS.batch_size)
    if dataset.data_size() % FLAGS.batch_size > 0:
        t_batches += 1
    # v_batches = int(dataset.data_size() / FLAGS.batch_size)
    # if val_data.data_size() % FLAGS.batch_size > 0:
    #     v_batches += 1

    with tf.Session() as sess:
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
                # Pull the image samples we'll use for training.
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
            # print("------------------------")
            # print(" Start validation >>> ")
            # print("------------------------")
            # # Reinitialize iterator with the validation dataset
            #
            # total_val_accuracy = 0
            # validation_count = 0
            # total_conf_matrix = None
            # for i in range(val_batches_per_epoch):
            #     validation_batch_xs, validation_batch_ys = sess.run(next_batch)
            #     # Run a validation step and capture training summaries for TensorBoard
            #     # with the `merged` op.
            #     validation_summary, validation_accuracy, conf_matrix = sess.run(
            #         [merged_summaries, accuracy, confusion_matrix],
            #         feed_dict={
            #             X: validation_batch_xs,
            #             ground_truth: validation_batch_ys,
            #             phase_train: False,
            #             # dropout_prob: 1.0
            #         })
            #
            #     validation_writer.add_summary(validation_summary, training_epoch)
            #
            #     total_val_accuracy += validation_accuracy
            #     validation_count += 1
            #     if total_conf_matrix is None:
            #         total_conf_matrix = conf_matrix
            #     else:
            #         total_conf_matrix += conf_matrix
            #
            # total_val_accuracy /= validation_count
            #
            # tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
            # tf.logging.info('Validation accuracy = %.1f%% (N=%d)' %
            #                 (total_val_accuracy * 100, raw_data.get_size('validation')))
            #
            # # Save the model checkpoint periodically.
            # if (training_epoch % FLAGS.save_step_interval == 0 or training_epoch == training_epochs_max):
            #     checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.model_architecture + '.ckpt')
            #     tf.logging.info('Saving to "%s-%d"', checkpoint_path, training_epoch)
            #     saver.save(sess, checkpoint_path, global_step=training_epoch)


if __name__ == '__main__':
    tf.app.run()
