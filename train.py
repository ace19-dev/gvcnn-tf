import os
import cv2

import tensorflow as tf

import numpy as np

import train_data
import val_data
from nets import model
from utils import train_utils, train_helper

slim = tf.contrib.slim

flags = tf.app.flags
FLAGS = flags.FLAGS


# Settings for logging.
flags.DEFINE_string('train_logdir', './tfmodels',
                    'Where the checkpoint and logs are stored.')
flags.DEFINE_string('ckpt_name_to_save', 'gvcnn.ckpt',
                    'Name to save checkpoint file')
flags.DEFINE_integer('log_steps', 10,
                     'Display logging information at every log_steps.')
flags.DEFINE_integer('save_interval_secs', 1200,
                     'How often, in seconds, we save the model to disk.')
flags.DEFINE_boolean('save_summaries_images', False,
                     'Save sample inputs, labels, and semantic predictions as '
                     'images to summary.')
flags.DEFINE_string('summaries_dir', './tfmodels/train_logs',
                     'Where to save summary logs for TensorBoard.')

flags.DEFINE_enum('learning_policy', 'poly', ['poly', 'step'],
                  'Learning rate policy for training.')
flags.DEFINE_float('base_learning_rate', .001,
                   'The base learning rate for model training.')
flags.DEFINE_float('learning_rate_decay_factor', 1e-3,
                   'The rate to decay the base learning rate.')
flags.DEFINE_float('learning_rate_decay_step', .3000,
                   'Decay the base learning rate at a fixed step.')
flags.DEFINE_float('learning_power', 0.9,
                   'The power value used in the poly learning policy.')
flags.DEFINE_float('training_number_of_steps', 300000,
                   'The number of steps used for training.')
flags.DEFINE_float('momentum', 0.9, 'The momentum value to use')

flags.DEFINE_float('last_layer_gradient_multiplier', 1.0,
                   'The gradient multiplier for last layers, which is used to '
                   'boost the gradient of last layers if the value > 1.')
# Set to False if one does not want to re-use the trained classifier weights.
flags.DEFINE_boolean('initialize_last_layer', True,
                     'Initialize the last layer.')
flags.DEFINE_boolean('last_layers_contain_logits_only', False,
                     'Only consider logits as last layers or not.')
flags.DEFINE_integer('slow_start_step', 0,
                     'Training model with small learning rate for few steps.')
flags.DEFINE_float('slow_start_learning_rate', 1e-4,
                   'Learning rate employed during slow start.')

# Settings for fine-tuning the network.
flags.DEFINE_string('saved_checkpoint_dir',
                    # './tfmodels',
                    None,
                    'Saved checkpoint dir.')
flags.DEFINE_string('pre_trained_checkpoint',
                    None,
                    'The pre-trained checkpoint in tensorflow format.')
flags.DEFINE_string('checkpoint_exclude_scopes',
                    None,
                    'Comma-separated list of scopes of variables to exclude '
                    'when restoring from a checkpoint.')
flags.DEFINE_string('trainable_scopes',
                    None,
                    'Comma-separated list of scopes to filter the set of variables '
                    'to train. By default, None would train all the variables.')
flags.DEFINE_string('checkpoint_model_scope',
                    None,
                    'Model scope in the checkpoint. None if the same as the trained model.')
flags.DEFINE_string('model_name',
                    'inception_v3',
                    'The name of the architecture to train.')
flags.DEFINE_boolean('ignore_missing_vars',
                     False,
                     'When restoring a checkpoint would ignore missing variables.')

# Dataset settings.
flags.DEFINE_string('dataset_dir', '/home/ace19/dl_data/modelnet2',
                    'Where the dataset reside.')

flags.DEFINE_integer('how_many_training_epochs', 100,
                     'How many training loops to runs')

# currently only use 1 batch size
# flags.DEFINE_integer('batch_size', 1, 'batch size')
# flags.DEFINE_integer('val_batch_size', 1, 'val batch size')
flags.DEFINE_integer('num_views', 6, 'number of views')
flags.DEFINE_integer('num_group', 10, 'number of group')
flags.DEFINE_integer('height', 299, 'height')
flags.DEFINE_integer('width', 299, 'width')
flags.DEFINE_string('labels',
                    # 'airplane,bed,bookshelf,bottle,chair,monitor,sofa,table,toilet,vase',
                    'monitor,toilet',
                    'number of classes')

# check total count before training
# MODELNET_TRAIN_DATA_SIZE = 626+515+572+335+889+465+680+392+344+475   # 5293, 10 class
# MODELNET_VALIDATE_DATA_SIZE = 1000
MODELNET_TRAIN_DATA_SIZE = 515+394    # 2 class
MODELNET_VALIDATE_DATA_SIZE = 100



def main(unused_argv):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    labels = FLAGS.labels.split(',')
    num_classes = len(labels)

    with tf.Graph().as_default() as graph:
        global_step = tf.compat.v1.train.get_or_create_global_step()

        # Define the model
        X = tf.compat.v1.placeholder(tf.float32,
                                     [None, FLAGS.num_views, FLAGS.height, FLAGS.width, 3],
                                     name='X')
        ground_truth = tf.compat.v1.placeholder(tf.int64, [None], name='ground_truth')
        is_training = tf.compat.v1.placeholder(tf.bool)
        dropout_keep_prob = tf.compat.v1.placeholder(tf.float32)
        g_scheme = tf.compat.v1.placeholder(tf.int32, [FLAGS.num_group, FLAGS.num_views])
        g_weight = tf.compat.v1.placeholder(tf.float32, [FLAGS.num_group,])

        # GVCNN
        view_scores, _, logits=model.gvcnn(X,
                                           num_classes,
                                           g_scheme,
                                           g_weight,
                                           is_training,
                                           dropout_keep_prob)

        # Define loss
        tf.losses.sparse_softmax_cross_entropy(labels=ground_truth, logits=logits)

        # Gather initial summaries.
        summaries = set(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES))

        prediction = tf.argmax(logits, 1, name='prediction')
        correct_prediction = tf.equal(prediction, ground_truth)
        confusion_matrix = tf.math.confusion_matrix(ground_truth,
                                                    prediction,
                                                    num_classes=num_classes)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        summaries.add(tf.compat.v1.summary.scalar('accuracy', accuracy))

        # # Add summaries for model variables.
        # for model_var in slim.get_model_variables():
        #     summaries.add(tf.compat.v1.summary.histogram(model_var.op.name, model_var))

        # Add summaries for losses.
        for loss in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.LOSSES):
            summaries.add(tf.compat.v1.summary.scalar('losses/%s' % loss.op.name, loss))

        learning_rate = train_utils.get_model_learning_rate(
            FLAGS.learning_policy, FLAGS.base_learning_rate,
            FLAGS.learning_rate_decay_step, FLAGS.learning_rate_decay_factor,
            FLAGS.training_number_of_steps, FLAGS.learning_power,
            FLAGS.slow_start_step, FLAGS.slow_start_learning_rate)
        optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate, FLAGS.momentum)
        summaries.add(tf.compat.v1.summary.scalar('learning_rate', learning_rate))

        total_loss, grads_and_vars = train_utils.optimize(optimizer)
        total_loss = tf.debugging.check_numerics(total_loss, 'Loss is inf or nan.')
        summaries.add(tf.compat.v1.summary.scalar('total_loss', total_loss))

        # Gather update_ops.
        # These contain, for example, the updates for the batch_norm variables created by model.
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)

        # Create gradient update op.
        update_ops.append(optimizer.apply_gradients(grads_and_vars,
                                                    global_step=global_step))
        update_op = tf.group(*update_ops)
        with tf.control_dependencies([update_op]):
            train_op = tf.identity(total_loss, name='train_op')


        ################
        # Prepare data
        ################
        filenames = tf.compat.v1.placeholder(tf.string, shape=[])
        tr_dataset = train_data.Dataset(filenames,
                                         FLAGS.num_views,
                                         FLAGS.height,
                                         FLAGS.width,
                                         1)  # batch_size
        iterator = tr_dataset.dataset.make_initializable_iterator()
        next_batch = iterator.get_next()

        # validation dateset
        val_dataset = val_data.Dataset(filenames,
                                        FLAGS.num_views,
                                        FLAGS.height,
                                        FLAGS.width,
                                        1)   # val_batch_size
        val_iterator = val_dataset.dataset.make_initializable_iterator()
        val_next_batch = val_iterator.get_next()

        sess_config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
        with tf.compat.v1.Session(config=sess_config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())

            # Add the summaries. These contain the summaries
            # created by model and either optimize() or _gather_loss().
            summaries |= set(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES))

            # Merge all summaries together.
            summary_op = tf.compat.v1.summary.merge(list(summaries))
            train_writer = tf.compat.v1.summary.FileWriter(FLAGS.summaries_dir, graph)
            validation_writer = tf.compat.v1.summary.FileWriter(FLAGS.summaries_dir + '/validation', graph)

            # Create a saver object which will save all the variables
            saver = tf.compat.v1.train.Saver(keep_checkpoint_every_n_hours=1.0)
            if FLAGS.pre_trained_checkpoint:
                train_utils.restore_fn(FLAGS)

            if FLAGS.saved_checkpoint_dir:
                if tf.gfile.IsDirectory(FLAGS.saved_checkpoint_dir):
                    checkpoint_path = tf.train.latest_checkpoint(FLAGS.saved_checkpoint_dir)
                else:
                    checkpoint_path = FLAGS.saved_checkpoint_dir
                saver.restore(sess, checkpoint_path)

            start_epoch = 0
            # Get the number of training/validation steps per epoch
            tr_batches = int(MODELNET_TRAIN_DATA_SIZE / 1)
            if MODELNET_TRAIN_DATA_SIZE % 1 > 0:
                tr_batches += 1
            val_batches = int(MODELNET_VALIDATE_DATA_SIZE / 1)
            if MODELNET_VALIDATE_DATA_SIZE % 1 > 0:
                val_batches += 1

            # The filenames argument to the TFRecordDataset initializer can either be a string,
            # a list of strings, or a tf.Tensor of strings.
            training_filenames = os.path.join(FLAGS.dataset_dir, 'modelnet2_6view_train.record')
            validate_filenames = os.path.join(FLAGS.dataset_dir, 'modelnet2_6view_test.record')

            ###################################
            # Training loop.
            ###################################
            for num_epoch in range(start_epoch, FLAGS.how_many_training_epochs):
                print("-------------------------------------")
                print(" Epoch {} ".format(num_epoch))
                print("-------------------------------------")

                sess.run(iterator.initializer, feed_dict={filenames: training_filenames})
                for step in range(tr_batches):
                    # Pull the image batch we'll use for training.
                    train_batch_xs, train_batch_ys = sess.run(next_batch)

                    # Sets up a graph with feeds and fetches for partial run.
                    handle = sess.partial_run_setup([view_scores, learning_rate,
                                                     # summary_op, top1_acc, loss, optimize_op, dummy],
                                                    summary_op, accuracy, loss, train_op],
                                                    [X, is_training, dropout_keep_prob,
                                                     ground_truth, g_scheme, g_weight])

                    _view_scores = sess.partial_run(handle,
                                                    [view_scores],
                                                     feed_dict={X: train_batch_xs,
                                                                ground_truth: train_batch_ys,
                                                                is_training: True,
                                                                dropout_keep_prob: 0.8}
                                                     )
                    _g_schemes = model.group_scheme(_view_scores, FLAGS.num_group, FLAGS.num_views)
                    _g_weights = model.group_weight(_g_schemes)

                    # Run the graph with this batch of training data.
                    lr, train_summary, train_accuracy, train_loss, _ = \
                        sess.partial_run(handle,
                                         # [learning_rate, summary_op, accuracy, loss, dummy],
                                         [learning_rate, summary_op, accuracy, loss, train_op],
                                         feed_dict={
                                             g_scheme: _g_schemes,
                                             g_weight: _g_weights}
                                         )

                    # # for debug
                    # lr, train_summary, train_accuracy, train_loss, _ = \
                    #     sess.run(# [learning_rate, summary_op, accuracy, loss, dummy],
                    #                      [learning_rate, summary_op, accuracy, loss, train_op],
                    #                      feed_dict={
                    #                          X: train_batch_xs,
                    #                          ground_truth: train_batch_ys,
                    #                          is_training: True,
                    #                          dropout_keep_prob: 0.8}
                    #                      )

                    train_writer.add_summary(train_summary, num_epoch)
                    tf.compat.v1.logging.info('Epoch #%d, Step #%d, rate %.6f, top1_acc %.3f%%, loss %.5f' %
                                    (num_epoch, step, lr, train_accuracy, train_loss))


                ###################################################
                # Validate the model on the validation set
                ###################################################
                tf.compat.v1.logging.info('--------------------------')
                tf.compat.v1.logging.info(' Start validation ')
                tf.compat.v1.logging.info('--------------------------')

                total_val_losses = 0.0
                total_val_top1_acc = 0.0
                val_count = 0
                total_conf_matrix = None

                # Reinitialize val_iterator with the validation dataset
                sess.run(val_iterator.initializer, feed_dict={filenames: validate_filenames})
                for step in range(val_batches):
                    validation_batch_xs, validation_batch_ys = sess.run(val_next_batch)

                    # Sets up a graph with feeds and fetches for partial run.
                    handle = sess.partial_run_setup([view_scores, summary_op,
                                                     accuracy, loss, confusion_matrix],
                                                    [X, is_training, dropout_keep_prob,
                                                     ground_truth, g_scheme, g_weight])

                    _view_scores = sess.partial_run(handle,
                                                    [view_scores],
                                                     feed_dict={X: validation_batch_xs,
                                                                ground_truth: validation_batch_ys,
                                                                is_training: False,
                                                                dropout_keep_prob: 1.0}
                                                     )
                    _g_schemes = model.group_scheme(_view_scores, FLAGS.num_group, FLAGS.num_views)
                    _g_weights = model.group_weight(_g_schemes)

                    # Run the graph with this batch of training data.
                    val_summary, val_accuracy, val_loss, conf_matrix = \
                        sess.partial_run(handle,
                                         [summary_op, accuracy, loss, confusion_matrix],
                                         feed_dict={
                                             g_scheme: _g_schemes,
                                             g_weight: _g_weights}
                                         )

                    # # for debug
                    # val_summary, val_accuracy, val_loss, conf_matrix = \
                    #     sess.run(# [learning_rate, summary_op, accuracy, loss, dummy],
                    #                      [summary_op, accuracy, loss, confusion_matrix],
                    #                      feed_dict={
                    #                          X: validation_batch_xs,
                    #                          ground_truth: validation_batch_ys,
                    #                          is_training: False,
                    #                          dropout_keep_prob: 1.0}
                    #                      )

                    validation_writer.add_summary(val_summary, num_epoch)

                    total_val_losses += val_loss
                    total_val_top1_acc += val_accuracy
                    val_count += 1
                    if total_conf_matrix is None:
                        total_conf_matrix = conf_matrix
                    else:
                        total_conf_matrix += conf_matrix

                total_val_losses /= val_count
                total_val_top1_acc /= val_count

                tf.compat.v1.logging.info('Confusion Matrix:\n %s' % total_conf_matrix)
                tf.compat.v1.logging.info('Validation loss = %.5f' % total_val_losses)
                tf.compat.v1.logging.info('Validation accuracy = %.3f%% (N=%d)' %
                                (total_val_top1_acc, MODELNET_VALIDATE_DATA_SIZE))

                # Save the model checkpoint periodically.
                if (num_epoch <= FLAGS.how_many_training_epochs-1):
                    checkpoint_path = os.path.join(FLAGS.train_logdir, FLAGS.ckpt_name_to_save)
                    tf.compat.v1.logging.info('Saving to "%s-%d"', checkpoint_path, num_epoch)
                    saver.save(sess, checkpoint_path, global_step=num_epoch)


if __name__ == '__main__':
    tf.compat.v1.logging.info('Creating train logdir: %s', FLAGS.train_logdir)
    tf.io.gfile.makedirs(FLAGS.train_logdir)

    tf.compat.v1.app.run()
