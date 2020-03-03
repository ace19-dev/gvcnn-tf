# Given C classes in the classification task,
# the output of the last layer in our network architecture is a
# vector with C elements, i.e., V = {v 1 , v 2 , · · · , v C }. Each
# element represents the probability that the subject belongs
# to that category. And the category with the largest value is
# the category it belongs to


import datetime
import os

import tensorflow as tf

import eval_data
from nets import model

slim = tf.contrib.slim

flags = tf.app.flags
FLAGS = flags.FLAGS


NUM_GROUP = 10

# temporary constant
MODELNET_EVAL_DATA_SIZE = 150


# Dataset settings.
flags.DEFINE_string('dataset_path', '/home/ace19/dl_data/modelnet/test.record',
                    'Where the dataset reside.')

flags.DEFINE_string('checkpoint_path',
                    os.getcwd() + '/models',
                    'Directory where to read training checkpoints.')

flags.DEFINE_integer('batch_size', 4, 'batch size')
flags.DEFINE_integer('num_views', 6, 'number of views')
flags.DEFINE_integer('height', 299, 'height')
flags.DEFINE_integer('width', 299, 'width')
flags.DEFINE_string('labels',
                    'airplane,bed,bookshelf,toilet,vase',
                    'number of classes')

def main(unused_argv):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    labels = FLAGS.labels.split(',')
    num_classes = len(labels)

    # Define the model
    X = tf.compat.v1.placeholder(tf.float32,
                                 [None, FLAGS.num_views, FLAGS.height, FLAGS.width, 3],
                                 name='X')
    # final_X = tf.compat.v1.placeholder(tf.float32,
    #                          [FLAGS.num_views, None, 8, 8, 1536],
    #                          name='final_X')
    ground_truth = tf.compat.v1.placeholder(tf.int64, [None], name='ground_truth')
    is_training = tf.compat.v1.placeholder(tf.bool, name='is_training')
    dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name='dropout_keep_prob')
    # grouping_scheme = tf.placeholder(tf.bool, [NUM_GROUP, FLAGS.num_views])
    # grouping_weight = tf.placeholder(tf.float32, [NUM_GROUP, 1])
    g_scheme = tf.compat.v1.placeholder(tf.int32, [FLAGS.num_group, FLAGS.num_views])
    g_weight = tf.compat.v1.placeholder(tf.float32, [FLAGS.num_group])

    # # Grouping Module
    # d_scores, _, final_desc = model.discrimination_score(X,
    #                                                      num_classes,
    #                                                      is_training)

    # # GVCNN
    # logits, _ = model.gvcnn(final_X,
    #                         grouping_scheme,
    #                         grouping_weight,
    #                         num_classes,
    #                         is_training2,
    #                         dropout_keep_prob)

    # GVCNN
    view_scores, _, logits = model.gvcnn(X,
                                         num_classes,
                                         g_scheme,
                                         g_weight,
                                         is_training,
                                         dropout_keep_prob)

    # prediction = tf.nn.softmax(logits)
    # predicted_labels = tf.argmax(prediction, 1)

    # prediction = tf.argmax(logits, 1, name='prediction')
    # correct_prediction = tf.equal(prediction, ground_truth)
    # confusion_matrix = tf.confusion_matrix(
    #     ground_truth, prediction, num_classes=num_classes)
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    prediction = tf.argmax(logits, 1, name='prediction')
    correct_prediction = tf.equal(prediction, ground_truth)
    confusion_matrix = tf.math.confusion_matrix(ground_truth,
                                                prediction,
                                                num_classes=num_classes)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    ################
    # Prepare data
    ################
    filenames = tf.compat.v1.placeholder(tf.string, shape=[])
    eval_dataset = eval_data.Dataset(filenames,
                                     FLAGS.num_views,
                                     FLAGS.height,
                                     FLAGS.width,
                                     FLAGS.batch_size)
    iterator = eval_dataset.dataset.make_initializable_iterator()
    next_batch = iterator.get_next()

    sess_config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
    with tf.compat.v1.Session(config=sess_config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        # Create a saver object which will save all the variables
        saver = tf.compat.v1.train.Saver()
        if FLAGS.checkpoint_path:
            if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
                checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            else:
                checkpoint_path = FLAGS.checkpoint_path
            saver.restore(sess, checkpoint_path)

        # global_step = checkpoint_path.split('/')[-1].split('-')[-1]

        # Get the number of training/validation steps per epoch
        batches = int(MODELNET_EVAL_DATA_SIZE / FLAGS.batch_size)
        if MODELNET_EVAL_DATA_SIZE % FLAGS.batch_size > 0:
            batches += 1

        ##############
        # prediction
        ##############
        start_time = datetime.datetime.now()
        tf.logging.info("Start prediction: %s" % start_time)

        eval_filenames = os.path.join(FLAGS.dataset_path)
        sess.run(iterator.initializer, feed_dict={filenames: eval_filenames})

        count = 0;
        total_acc = 0
        total_conf_matrix = None
        for i in range(batches):
            batch_xs, batch_ys, _ = sess.run(next_batch)

            # # Sets up a graph with feeds and fetches for partial runs.
            # handle = sess.partial_run_setup([d_scores, final_desc,
            #                                  accuracy, confusion_matrix],
            #                                 [X, final_X, ground_truth,
            #                                  grouping_scheme, grouping_weight, is_training,
            #                                  is_training2, dropout_keep_prob])
            #
            # scores, final = sess.partial_run(handle,
            #                                  [d_scores, final_desc],
            #                                  feed_dict={
            #                                      X: batch_xs,
            #                                      is_training: False}
            #                                  )
            # schemes = model.grouping_scheme(scores, NUM_GROUP, FLAGS.num_views)
            # weights = model.grouping_weight(scores, schemes)
            #
            # # Run the graph with this batch of training data.
            # acc, conf_matrix  = \
            #     sess.partial_run(handle,
            #                      [accuracy, confusion_matrix],
            #                      feed_dict={
            #                          final_X: final,
            #                          ground_truth: batch_ys,
            #                          grouping_scheme: schemes,
            #                          grouping_weight: weights,
            #                          is_training2: False,
            #                          dropout_keep_prob: 1.0}
            #                      )


            # Sets up a graph with feeds and fetches for partial run.
            handle = sess.partial_run_setup([view_scores, accuracy, confusion_matrix],
                                            [X, g_scheme, g_weight,
                                             ground_truth, is_training, dropout_keep_prob])

            _view_scores = sess.partial_run(handle,
                                            [view_scores],
                                            feed_dict={
                                                X: batch_xs,
                                                is_training: False,
                                                dropout_keep_prob: 1.0}
                                            )
            _g_schemes = model.group_scheme(_view_scores, FLAGS.num_group, FLAGS.num_views)
            _g_weights = model.group_weight(_g_schemes)

            # Run the graph with this batch of training data.
            acc, conf_matrix = \
                sess.partial_run(handle,
                                 [accuracy, confusion_matrix],
                                 feed_dict={
                                     ground_truth: batch_ys,
                                     g_scheme: _g_schemes,
                                     g_weight: _g_weights}
                                 )

            total_acc += acc
            count += 1

            if total_conf_matrix is None:
                total_conf_matrix = conf_matrix
            else:
                total_conf_matrix += conf_matrix

        total_acc /= count
        tf.compat.v1.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
        tf.compat.v1.logging.info('Final test accuracy = %.3f%% (N=%d)' %
                                  (total_acc * 100, MODELNET_EVAL_DATA_SIZE))

        end_time = datetime.datetime.now()
        tf.compat.v1.logging.info('End prediction: %s' % end_time)
        tf.compat.v1.logging.info('prediction waste time: %s' % (end_time - start_time))


if __name__ == '__main__':
    tf.app.run()
