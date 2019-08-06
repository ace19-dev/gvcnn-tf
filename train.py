import os
import cv2
from random import randrange

import tensorflow as tf
from tensorflow.python.client import device_lib

import numpy as np

import train_data
import val_data
import gvcnn
from utils import train_utils, train_helper

slim = tf.contrib.slim

flags = tf.app.flags
FLAGS = flags.FLAGS


# Multi GPU
flags.DEFINE_integer('num_gpu', 4, 'number of GPU')

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
flags.DEFINE_float('learning_rate_decay_step', .2000,
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
                    # './pre-trained/inception_v4.ckpt',
                    None,
                    'The pre-trained checkpoint in tensorflow format.')
flags.DEFINE_string('checkpoint_exclude_scopes',
                    # 'gvcnn/AuxLogits,gvcnn/Logits',
                    None,
                    'Comma-separated list of scopes of variables to exclude '
                    'when restoring from a checkpoint.')
flags.DEFINE_string('trainable_scopes',
                    # 'gvcnn/AuxLogits, gvcnn/Logits',
                    None,
                    'Comma-separated list of scopes to filter the set of variables '
                    'to train. By default, None would train all the variables.')
flags.DEFINE_string('checkpoint_model_scope',
                    None,
                    'Model scope in the checkpoint. None if the same as the trained model.')
flags.DEFINE_string('model_name',
                    'inception_v4',
                    'The name of the architecture to train.')
flags.DEFINE_boolean('ignore_missing_vars',
                     False,
                     'When restoring a checkpoint would ignore missing variables.')

# Dataset settings.
flags.DEFINE_string('dataset_dir', '/home/ace19/dl_data/modelnet12',
                    'Where the dataset reside.')

flags.DEFINE_integer('how_many_training_epochs', 100,
                     'How many training loops to run')

# TODO: batch size must be multiple of num_gpu
flags.DEFINE_integer('batch_size', 1, 'batch size')
flags.DEFINE_integer('val_batch_size', 1, 'val batch size')
flags.DEFINE_integer('num_views', 6, 'number of views')
flags.DEFINE_integer('height', 299, 'height')
flags.DEFINE_integer('width', 299, 'width')
flags.DEFINE_string('labels',
                    'airplane,bed,bookshelf,bottle,chair,monitor,piano,plant,sofa,table,toilet,vase',
                    'number of classes')


# relate to grouping_scheme func.
NUM_GROUP = 12

MODELNET_TRAIN_DATA_SIZE = 5764
MODELNET_VALIDATE_DATA_SIZE = 1200


def get_filenames(dataset_category):
    filenames = []

    tfrecord_files = os.listdir(FLAGS.dataset_dir)
    for f in tfrecord_files:
        if dataset_category in f:
            filenames.append(os.path.join(FLAGS.dataset_dir, f))

    return filenames

def show_batch_data(step, batch_x, batch_y, additional_path=None):
    default_path = '/home/ace19/Pictures/'
    if additional_path is not None:
        default_path = os.path.join(default_path, additional_path)
        if not os.path.exists(default_path):
            os.makedirs(default_path)

    assert not np.any(np.isnan(batch_x))

    n_batch = batch_x.shape[0]
    # n_view = batch_x.shape[1]
    for i in range(n_batch):
        batch_imgs = batch_x[i]
        n_img = batch_imgs.shape[0]
        for k in range(n_img):
            img = batch_imgs[k]
            # scipy.misc.toimage(img).show() Or
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            cv2.imwrite(
                os.path.join(default_path, str(step) + '-step_' + str(batch_y[i]) + '-lable_' +
                             str(i) + '-batch_' + str(k) + '-image' + '.png'),
                img)
            # cv2.imshow(str(batch_y[idx]), img)
            cv2.waitKey(10)
            cv2.destroyAllWindows()


def main(unused_argv):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    # device = device_lib.list_local_devices()

    labels = FLAGS.labels.split(',')
    num_classes = len(labels)

    with tf.Graph().as_default() as graph:
        global_step = tf.compat.v1.train.get_or_create_global_step()

        # Define the model
        X = tf.compat.v1.placeholder(tf.float32,
                           [None, FLAGS.num_views, FLAGS.height, FLAGS.width, 3],
                           name='X')
        # for 299 size, otherwise you should modify shape for your size.
        final_X = tf.compat.v1.placeholder(tf.float32,
                                 [FLAGS.num_views, None, 8, 8, 1536],
                                 name='final_X')
        ground_truth = tf.compat.v1.placeholder(tf.int64, [None], name='ground_truth')
        is_training = tf.compat.v1.placeholder(tf.bool)
        is_training2 = tf.compat.v1.placeholder(tf.bool)
        dropout_keep_prob = tf.compat.v1.placeholder(tf.float32)
        grouping_scheme = tf.compat.v1.placeholder(tf.bool, [NUM_GROUP, FLAGS.num_views])
        grouping_weight = tf.compat.v1.placeholder(tf.float32, [NUM_GROUP, 1])

        # Gather initial summaries.
        summaries = set(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES))

        learning_rate = train_utils.get_model_learning_rate(
            FLAGS.learning_policy, FLAGS.base_learning_rate,
            FLAGS.learning_rate_decay_step, FLAGS.learning_rate_decay_factor,
            FLAGS.training_number_of_steps, FLAGS.learning_power,
            FLAGS.slow_start_step, FLAGS.slow_start_learning_rate)
        summaries.add(tf.compat.v1.summary.scalar('learning_rate', learning_rate))

        optimizers = \
            [tf.compat.v1.train.MomentumOptimizer(learning_rate, FLAGS.momentum) for _ in range(FLAGS.num_gpu)]

        logits = []
        losses = []
        grad_list = []
        filename_batch = []
        image_batch = []
        gt_batch = []
        for gpu_idx in range(FLAGS.num_gpu):
            tf.compat.v1.logging.info('creating gpu tower @ %d' % (gpu_idx + 1))
            image_batch.append(X)
            gt_batch.append(ground_truth)

            scope_name = 'tower%d' % gpu_idx
            with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.compat.v1.variable_scope(scope_name):
                # Grouping Module
                d_scores, _, final_desc = gvcnn.discrimination_score(X, num_classes, is_training)

                # GVCNN
                logit, _ = gvcnn.gvcnn(final_X,
                                       grouping_scheme,
                                       grouping_weight,
                                       num_classes,
                                       is_training2,
                                       dropout_keep_prob)

                logits.append(logit)

                l = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ground_truth,
                                                                   logits=logit)
                losses.append(l)
                loss_w_reg = tf.reduce_sum(l) + tf.add_n(tf.compat.v1.losses.get_regularization_losses(scope=scope_name))

                grad_list.append(
                    [x for x in optimizers[gpu_idx].compute_gradients(loss_w_reg) if x[0] is not None])

        y_hat = tf.concat(logits, axis=0)
        image_batch = tf.concat(image_batch, axis=0)
        gt_batch = tf.concat(gt_batch, axis=0)

        top1_acc = tf.reduce_mean(
            tf.cast(tf.nn.in_top_k(y_hat, gt_batch, k=1), dtype=tf.float32)
        )
        summaries.add(tf.compat.v1.summary.scalar('top1_acc', top1_acc))
        prediction = tf.argmax(y_hat, axis=1, name='prediction')
        confusion_matrix = tf.math.confusion_matrix(gt_batch,
                                                    prediction,
                                                    num_classes=num_classes)

        loss = tf.reduce_mean(losses)
        loss = tf.compat.v1.check_numerics(loss, 'Loss is inf or nan.')
        summaries.add(tf.compat.v1.summary.scalar('loss', loss))

        # use NCCL
        grads, all_vars = train_helper.split_grad_list(grad_list)
        reduced_grad = train_helper.allreduce_grads(grads, average=True)
        grads = train_helper.merge_grad_list(reduced_grad, all_vars)

        # optimizer using NCCL
        train_ops = []
        for idx, grad_and_vars in enumerate(grads):
            # apply_gradients may create variables. Make them LOCAL_VARIABLESZ¸¸¸¸¸¸
            with tf.name_scope('apply_gradients'), tf.device(tf.DeviceSpec(device_type="GPU", device_index=idx)):
                update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, scope='tower%d' % idx)
                with tf.control_dependencies(update_ops):
                    train_ops.append(
                        optimizers[idx].apply_gradients(grad_and_vars, name='apply_grad_{}'.format(idx),
                                                        global_step=global_step)
                    )
                # TODO:
                # TensorBoard: How to plot histogram for gradients
                # grad_summ_op = tf.summary.merge([tf.summary.histogram("%s-grad" % g[1].name, g[0]) for g in grads_and_vars])
        optimize_op = tf.group(*train_ops, name='train_op')
        # https://github.com/tensorflow/tensorflow/issues/1899
        # TODO: check
        with tf.control_dependencies([optimize_op]):
            dummy = tf.constant(0)

        sync_op = train_helper.get_post_init_ops()

        ################
        # Prepare data
        ################
        filenames = tf.compat.v1.placeholder(tf.string, shape=[])
        tr_dataset = train_data.Dataset(filenames,
                                        FLAGS.num_views,
                                        FLAGS.height,
                                        FLAGS.width,
                                        FLAGS.batch_size)
        iterator = tr_dataset.dataset.make_initializable_iterator()
        next_batch = iterator.get_next()

        # validation dateset
        val_dataset = val_data.Dataset(filenames,
                                       FLAGS.num_views,
                                       FLAGS.height,
                                       FLAGS.width,
                                       FLAGS.val_batch_size)
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

            # TODO:
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

            sess.run(sync_op)

            start_epoch = 0
            # Get the number of training/validation steps per epoch
            tr_batches = int(MODELNET_TRAIN_DATA_SIZE / (FLAGS.batch_size))
            if MODELNET_TRAIN_DATA_SIZE % (FLAGS.batch_size) > 0:
                tr_batches += 1
            val_batches = int(MODELNET_VALIDATE_DATA_SIZE / (FLAGS.val_batch_size))
            if MODELNET_VALIDATE_DATA_SIZE % (FLAGS.val_batch_size) > 0:
                val_batches += 1

            # The filenames argument to the TFRecordDataset initializer can either be a string,
            # a list of strings, or a tf.Tensor of strings.
            training_filenames = os.path.join(FLAGS.dataset_dir, 'modelnet2_train.record')
            validate_filenames = os.path.join(FLAGS.dataset_dir, 'modelnet2_test.record')
            # training_filenames = get_filenames('train')
            # validate_filenames = get_filenames('test')
            ##################
            # Training loop.
            ##################
            for training_epoch in range(start_epoch, FLAGS.how_many_training_epochs):
                print("-------------------------------------")
                print(" Epoch {} ".format(training_epoch))
                print("-------------------------------------")

                sess.run(iterator.initializer, feed_dict={filenames: training_filenames})
                for step in range(tr_batches):
                    # index = randrange(num_classes)
                    # sess.run(iterator.initializer, feed_dict={filenames: training_filenames[index]})
                    # Pull the image batch we'll use for training.
                    train_batch_xs, train_batch_ys = sess.run(next_batch)
                    # show_batch_data(step, train_batch_xs, train_batch_ys)

                    # Sets up a graph with feeds and fetches for partial run.
                    handle = sess.partial_run_setup([d_scores, final_desc, learning_rate, summary_op,
                                                     top1_acc, loss, optimize_op, dummy],
                                                    [X, final_X, ground_truth,
                                                     grouping_scheme, grouping_weight, is_training,
                                                     is_training2, dropout_keep_prob])

                    scores, final = sess.partial_run(handle,
                                                     [d_scores, final_desc],
                                                     feed_dict={
                                                        X: train_batch_xs,
                                                        is_training: True}
                                                     )
                    schemes = gvcnn.grouping_scheme(scores, NUM_GROUP, FLAGS.num_views)
                    weights = gvcnn.grouping_weight(scores, schemes)

                    # Run the graph with this batch of training data.
                    lr, train_summary, train_accuracy, train_loss, _ = \
                        sess.partial_run(handle,
                                         [learning_rate, summary_op, top1_acc, loss, dummy],
                                         feed_dict={
                                             final_X: final,
                                             ground_truth: train_batch_ys,
                                             grouping_scheme: schemes,
                                             grouping_weight: weights,
                                             is_training2: True,
                                             dropout_keep_prob: 0.8}
                                         )

                    train_writer.add_summary(train_summary, training_epoch)
                    tf.compat.v1.logging.info('Epoch #%d, Step #%d, rate %.6f, top1_acc %.3f%%, loss %.5f' %
                                    (training_epoch, step, lr, train_accuracy, train_loss))


                ###################################################
                # Validate the model on the validation set
                ###################################################
                tf.compat.v1.logging.info('--------------------------')
                tf.compat.v1.logging.info(' Start validation ')
                tf.compat.v1.logging.info('--------------------------')

                # total_val_losses = 0.0
                total_val_top1_acc = 0.0
                val_count = 0
                total_conf_matrix = None

                # Reinitialize val_iterator with the validation dataset
                sess.run(val_iterator.initializer, feed_dict={filenames: validate_filenames})
                for step in range(val_batches):
                    # index = randrange(num_classes)
                    # sess.run(val_iterator.initializer, feed_dict={filenames: validate_filenames[index]})
                    validation_batch_xs, validation_batch_ys = sess.run(val_next_batch)
                    # show_batch_data(step, validation_batch_xs, validation_batch_ys)

                    # Sets up a graph with feeds and fetches for partial run.
                    handle = sess.partial_run_setup([d_scores, final_desc, summary_op,
                                                     top1_acc, confusion_matrix],
                                                    [X, final_X, ground_truth,
                                                     grouping_scheme, grouping_weight, is_training,
                                                     is_training2, dropout_keep_prob])

                    scores, final = sess.partial_run(handle,
                                                     [d_scores, final_desc],
                                                     feed_dict={
                                                         X: validation_batch_xs,
                                                         is_training: False}
                                                     )
                    schemes = gvcnn.grouping_scheme(scores, NUM_GROUP, FLAGS.num_views)
                    weights = gvcnn.grouping_weight(scores, schemes)

                    # Run the graph with this batch of training data.
                    val_summary, val_accuracy, conf_matrix = \
                        sess.partial_run(handle,
                                         [summary_op, top1_acc, confusion_matrix],
                                         feed_dict={
                                             final_X: final,
                                             ground_truth: validation_batch_ys,
                                             grouping_scheme: schemes,
                                             grouping_weight: weights,
                                             is_training2: False,
                                             dropout_keep_prob: 1.0}
                                         )

                    validation_writer.add_summary(val_summary, training_epoch)

                    # total_val_losses += val_loss
                    total_val_top1_acc += val_accuracy
                    val_count += 1
                    if total_conf_matrix is None:
                        total_conf_matrix = conf_matrix
                    else:
                        total_conf_matrix += conf_matrix

                # total_val_losses /= val_count
                total_val_top1_acc /= val_count

                tf.compat.v1.logging.info('Confusion Matrix:\n %s' % total_conf_matrix)
                # tf.compat.v1.logging.info('Validation loss = %.5f' % total_val_losses)
                tf.compat.v1.logging.info('Validation accuracy = %.3f%% (N=%d)' %
                                (total_val_top1_acc, MODELNET_VALIDATE_DATA_SIZE))

                # periodic synchronization
                sess.run(sync_op)

                # Save the model checkpoint periodically.
                if (training_epoch <= FLAGS.how_many_training_epochs-1):
                    checkpoint_path = os.path.join(FLAGS.train_logdir, FLAGS.ckpt_name_to_save)
                    tf.compat.v1.logging.info('Saving to "%s-%d"', checkpoint_path, training_epoch)
                    saver.save(sess, checkpoint_path, global_step=training_epoch)


if __name__ == '__main__':
    tf.compat.v1.logging.info('Creating train logdir: %s', FLAGS.train_logdir)
    tf.io.gfile.makedirs(FLAGS.train_logdir)

    tf.compat.v1.app.run()
