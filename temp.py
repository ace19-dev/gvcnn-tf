# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import csv
import datetime
import shutil

import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

from six.moves import xrange

import model

from input_data import Data
from input_data import DataLoader

from tensorflow.python.platform import gfile

FLAGS = None


def main(_):
    # We want to see all the logging messages for this tutorial.
    tf.logging.set_verbosity(tf.logging.INFO)

    labels = FLAGS.labels.split(',')
    label_count = len(labels)

    # Figure out the learning rates for each training phase. Since it's often
    # effective to have high learning rates at the start of training, followed by
    # lower levels towards the end, the number of steps and learning rates can be
    # specified as comma-separated lists to define the rate at each stage. For
    # example --how_many_training_epochs=10000,3000 --learning_rate=0.001,0.0001
    # will run 13,000 training loops in total, with a rate of 0.001 for the first
    # 10,000, and 0.0001 for the final 3,000.
    training_epochs_list = list(map(int, FLAGS.how_many_training_epochs.split(',')))
    learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
    if len(training_epochs_list) != len(learning_rates_list):
        raise Exception(
            '--how_many_training_epochs and --learning_rate must be equal length '
            'lists, but are %d and %d long instead' % (len(training_epochs_list),
                                                       len(learning_rates_list)))

    phase_train = tf.placeholder(tf.bool, name='phase_train')
    X = tf.placeholder(tf.float32,
                       [None, FLAGS.image_height, FLAGS.image_width, 3],
                       name='input')
    ground_truth = tf.placeholder(tf.int64, [None], name='ground_truth')

    logits, end_points = model.create_model(
        X,
        label_count,
        FLAGS.model_architecture,
        is_training=phase_train)
    # yhat = tf.identity(end_points['Predictions'], name='output') # use tf.identity() to assign name
    # logits = tf.identity(end_points['vgg_16/fc8'], name='output')
    logits = tf.identity(logits, name='output')

    # Define loss and optimizer
    with tf.name_scope('cross_entropy'):
        cross_entropy_mean = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
            labels=ground_truth, logits=logits))
    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    learning_rate_input = tf.placeholder(tf.float32, [], name='learning_rate_input')
    # momentum = tf.placeholder(tf.float32, [], name='momentum')
    # dropout_prob = tf.placeholder(tf.float32, [], name='dropout_prob')

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.GradientDescentOptimizer(learning_rate_input).minimize(cross_entropy_mean)
        # train_step = tf.train.MomentumOptimizer(learning_rate_input, 0.99, use_nesterov=True).minimize(cross_entropy_mean)
        # train_step = tf.train.AdamOptimizer(learning_rate_input).minimize(cross_entropy_mean)
        # train_step = tf.train.AdadeltaOptimizer(learning_rate_input).minimize(cross_entropy_mean)
        # train_step = tf.train.RMSPropOptimizer(learning_rate_input, 0.99).minimize(cross_entropy_mean)

    # Transform output to topK result.
    # values, indices = tf.nn.top_k(yhat, label_count, name='values')
    # table = tf.contrib.lookup.index_to_string_table_from_tensor(
    #     tf.constant([label for label in labels]))
    # prediction_classes = table.lookup(tf.to_int64(indices), name='pred_classes')

    prediction = tf.argmax(logits, 1, name='prediction')
    correct_prediction = tf.equal(prediction, ground_truth)
    confusion_matrix = tf.confusion_matrix(
        ground_truth, prediction, num_classes=label_count)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    global_step = tf.train.get_or_create_global_step()
    increment_global_step = tf.assign(global_step, global_step + 1)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
    merged_summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')

    # Create a saver object which will save all the variables
    saver = tf.train.Saver()

    start_epoch = 1
    start_checkpoint_epoch = 0
    if FLAGS.start_checkpoint:
        # model.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
        saver.restore(sess, FLAGS.start_checkpoint)
        tmp = FLAGS.start_checkpoint
        tmp = tmp.split('-')
        tmp.reverse()
        start_checkpoint_epoch = int(tmp[0])
        start_epoch = start_checkpoint_epoch + 1

    # calculate training epochs max
    training_epochs_max = np.sum(training_epochs_list)
    if start_checkpoint_epoch != training_epochs_max:
        tf.logging.info('Training from epoch: %d ', start_epoch)

    # Saving as Protocol Buffer (pb)
    tf.train.write_graph(sess.graph_def, FLAGS.train_dir,
                         FLAGS.model_architecture + '.pbtxt',
                         as_text=True)


    ############################
    # Prepare data
    ############################
    # Place data loading and preprocessing on the cpu
    # with tf.device('/cpu:0'):
    raw_data = Data(FLAGS.data_dir,
                    labels,
                    FLAGS.validation_percentage,
                    FLAGS.testing_percentage)

    tr_data = DataLoader(raw_data.get_data('training'),
                         raw_data.get_label_to_index(),
                         FLAGS.batch_size)

    val_data = DataLoader(raw_data.get_data('validation'),
                          raw_data.get_label_to_index(),
                          FLAGS.batch_size)

    # te_data = DataLoader(raw_data.get_data('testing'),
    #                      raw_data.get_label_to_index(),
    #                      FLAGS.batch_size)

    # create an reinitializable iterator given the dataset structure
    iterator = tf.data.Iterator.from_structure(tr_data.dataset.output_types,
                                               tr_data.dataset.output_shapes)
    next_batch = iterator.get_next()

    # Ops for initializing the two different iterators
    training_init_op = iterator.make_initializer(tr_data.dataset)
    validation_init_op = iterator.make_initializer(val_data.dataset)
    # testing_init_op = iterator.make_initializer(te_data.dataset)

    # Save list of words.
    with gfile.GFile(
            os.path.join(FLAGS.train_dir, FLAGS.model_architecture + '_labels.txt'),
            'w') as f:
        f.write('\n'.join(raw_data.labels_list))

    # Get the number of training/validation steps per epoch
    tr_batches_per_epoch = int(tr_data.data_size / FLAGS.batch_size)
    if tr_data.data_size % FLAGS.batch_size > 0:
        tr_batches_per_epoch += 1
    val_batches_per_epoch = int(val_data.data_size / FLAGS.batch_size)
    if val_data.data_size % FLAGS.batch_size > 0:
        val_batches_per_epoch += 1
    # te_batches_per_epoch = int(te_data.data_size / FLAGS.batch_size)
    # if te_data.data_size % FLAGS.batch_size > 0:
    #     te_batches_per_epoch += 1


    ############################
    # Training loop.
    ############################
    for training_epoch in xrange(start_epoch, training_epochs_max + 1):
        print("------------------------")
        print(" Epoch:{} >> ".format(training_epoch))
        print("------------------------")

        # Figure out what the current learning rate is.
        training_epochs_sum = 0
        for i in range(len(training_epochs_list)):
            training_epochs_sum += training_epochs_list[i]
            if training_epoch <= training_epochs_sum:
                learning_rate_value = learning_rates_list[i]
                break

        sess.run(training_init_op)  # Initialize iterator with the training dataset
        for step in range(tr_batches_per_epoch):
            # Pull the image samples we'll use for training.
            train_batch_xs, train_batch_ys = sess.run(next_batch)
            # for debugging
            # img = train_batch_xs[0]
            # plt.hist(img.ravel())
            # plt.show()

            # Run the graph with this batch of training data.
            train_summary, train_accuracy, cross_entropy_value, _, _ = \
                sess.run(
                    [
                        merged_summaries, accuracy, cross_entropy_mean, train_step,
                        increment_global_step
                    ],
                    feed_dict={
                        X: train_batch_xs,
                        ground_truth: train_batch_ys,
                        learning_rate_input: learning_rate_value,
                        phase_train: True,
                        # momentum: 0.95,
                        # dropout_prob: 0.5
                    }
                )

            train_writer.add_summary(train_summary, training_epoch)
            tf.logging.info('Epoch #%d, Step #%d, rate %f, accuracy %.1f%%, cross entropy %f' %
                            (training_epoch, step, learning_rate_value, train_accuracy * 100,
                             cross_entropy_value))

        ############################
        # Validate the model on the entire validation set
        ############################
        print("------------------------")
        print(" Start validation >>> ")
        print("------------------------")
        # Reinitialize iterator with the validation dataset
        sess.run(validation_init_op)
        total_val_accuracy = 0
        validation_count = 0
        total_conf_matrix = None
        for i in range(val_batches_per_epoch):
            validation_batch_xs, validation_batch_ys = sess.run(next_batch)
            # Run a validation step and capture training summaries for TensorBoard
            # with the `merged` op.
            validation_summary, validation_accuracy, conf_matrix = sess.run(
                [merged_summaries, accuracy, confusion_matrix],
                feed_dict={
                    X: validation_batch_xs,
                    ground_truth: validation_batch_ys,
                    phase_train: False,
                    # dropout_prob: 1.0
                })

            validation_writer.add_summary(validation_summary, training_epoch)

            total_val_accuracy += validation_accuracy
            validation_count += 1
            if total_conf_matrix is None:
                total_conf_matrix = conf_matrix
            else:
                total_conf_matrix += conf_matrix

        total_val_accuracy /= validation_count

        tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
        tf.logging.info('Validation accuracy = %.1f%% (N=%d)' %
                        (total_val_accuracy * 100, raw_data.get_size('validation')))

        # Save the model checkpoint periodically.
        if (training_epoch % FLAGS.save_step_interval == 0 or training_epoch == training_epochs_max):
            checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.model_architecture + '.ckpt')
            tf.logging.info('Saving to "%s-%d"', checkpoint_path, training_epoch)
            saver.save(sess, checkpoint_path, global_step=training_epoch)


    ############################
    # For Test
    ############################
    # start_time = datetime.datetime.now()
    # print("{} Start testing".format(start_time))
    # # Reinitialize iterator with the Evaluate dataset
    # sess.run(testing_init_op)
    #
    # total_test_accuracy = 0
    # test_count = 0
    # total_conf_matrix = None
    # for i in range(te_batches_per_epoch):
    #     test_batch_xs, test_batch_ys = sess.run(next_batch)
    #     test_accuracy, conf_matrix = sess.run(
    #         [accuracy, confusion_matrix],
    #         feed_dict={
    #             X: test_batch_xs,
    #             ground_truth: test_batch_ys,
    #             phase_train: False,
    #             # dropout_prob: 1.0
    #         })
    #
    #     total_test_accuracy += test_accuracy
    #     test_count += 1
    #
    #     if total_conf_matrix is None:
    #         total_conf_matrix = conf_matrix
    #     else:
    #         total_conf_matrix += conf_matrix
    #
    # total_test_accuracy /= test_count
    #
    # tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
    # tf.logging.info('Final test accuracy = %.1f%% (N=%d)' % (total_test_accuracy * 100,
    #                                                          raw_data.get_size('testing')))
    #
    # end_time = datetime.datetime.now()
    # print('End testing: ', end_time)
    # print('Testing waste time:{}'.format(end_time - start_time))


    # ############################
    # # export model
    # ############################
    # (re-)create export directory
    export_path = os.path.join(
        tf.compat.as_bytes(FLAGS.train_dir),
        tf.compat.as_bytes(FLAGS.model_architecture + str(FLAGS.model_version)))
    if os.path.exists(export_path):
        shutil.rmtree(export_path)

    # create model builder
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    predict_inputs_tensor_info = tf.saved_model.utils.build_tensor_info(X)
    predict_inputs_tensor_info2 = tf.saved_model.utils.build_tensor_info(phase_train)
    predict_tensor_classes_info = tf.saved_model.utils.build_tensor_info(prediction)
    # predict_tensor_scores_info = tf.saved_model.utils.build_tensor_info(values)
    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'images': predict_inputs_tensor_info,
                    'phase_train': predict_inputs_tensor_info2},
            outputs={'classes': predict_tensor_classes_info,},
                     # 'scores': predict_tensor_scores_info},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        ))

    # legacy_init_op = tf.group(
    #     tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        sess,
        [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict_images': prediction_signature,
        })
        # legacy_init_op=legacy_init_op)

    builder.save(as_text=True)
    print("Successfully exported model version '{}' into '{}'".format(
        FLAGS.model_version, export_path))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/home/ace19/dl_data/KGC/refined_train/',
        # default='/home/ace19/dl_data/skipc/train/',
        help="""\
        Where to download the image training data to.
        """)
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=0,
        help='What percentage of wavs to use as a test set.')
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help='What percentage of wavs to use as a validation set.')
    parser.add_argument(
        '--how_many_training_epochs',
        type=str,
        # default='10,20',
        default='100,50',
        help='How many training loops to run')
    parser.add_argument(
        '--eval_step_interval',
        type=int,
        default=1,
        help='How often to evaluate the training results.')
    parser.add_argument(
        '--learning_rate',
        type=str,
        # default='0.06,0.03',
        default='0.001,0.0001',
        help='How large a learning rate to use when training.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='How many items to train with at once', )
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default=os.getcwd() + '/models/train_logs',
        help='Where to save summary logs for TensorBoard.')
    parser.add_argument(
        '--labels',
        type=str,
        # default='Black-grass,Charlock,Cleavers,Common Chickweed,Common wheat,Fat Hen,Loose Silky-bent,Maize,Scentless Mayweed,Shepherds Purse,Small-flowered Cranesbill,Sugar beet',
        default='heaven,earth,good',
        help='Labels to use', )
    parser.add_argument(
        '--train_dir',
        type=str,
        default=os.getcwd() + '/models',
        help='Directory to write event logs and checkpoint.')
    parser.add_argument(
        '--save_step_interval',
        type=int,
        default=1,
        help='Save model checkpoint every save_steps.')
    parser.add_argument(
        '--start_checkpoint',
        type=str,
        # default=os.getcwd() + '/models/mobile.ckpt-20',
        default='',
        help='If specified, restore this pretrained model before any training.')
    parser.add_argument(
        '--model_architecture',
        type=str,
        default='resnet50',
        help='What model architecture to use')
    parser.add_argument(
        '--model_version',
        type=str,
        default='1.0',
        help='What model version to use')
    parser.add_argument(
        '--image_height',
        type=int,
        default=224,    # nasnet, mobilenet
        # default=299,
        help='how do you want image resize height.')
    parser.add_argument(
        '--image_width',
        type=int,
        default=224,  # nasnet, mobilenet
        # default=299,
        help='how do you want image resize width.')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
