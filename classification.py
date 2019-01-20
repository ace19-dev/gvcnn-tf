# Given C classes in the classification task,
# the output of the last layer in our network architecture is a
# vector with C elements, i.e., V = {v 1 , v 2 , · · · , v C }. Each
# element represents the probability that the subject belongs
# to that category. And the category with the largest value is
# the category it belongs to


import datetime
import os
import csv

import tensorflow as tf

import data
import gvcnn
from nets import inception_v4

slim = tf.contrib.slim

flags = tf.app.flags
FLAGS = flags.FLAGS


NUM_GROUP = 8

# temporary constant
MODELNET_EVAL_DATA_SIZE = 0


# Dataset settings.
flags.DEFINE_string('dataset_path', '/home/ace19/dl_data/modelnet/test.record',
                    'Where the dataset reside.')

# TODO: will apply n-batch later
flags.DEFINE_integer('batch_size', 1, 'batch size')
flags.DEFINE_integer('num_views', 8, 'number of views')
flags.DEFINE_integer('height', 299, 'height')
flags.DEFINE_integer('width', 299, 'width')
flags.DEFINE_string('labels',
                    'airplane,bed,bookshelf,cone,person,toilet,vase',
                    'number of classes')

def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    labels = FLAGS.labels.split(',')
    num_classes = len(labels)

    # Define the model
    X = tf.placeholder(tf.float32,
                       [None, FLAGS.num_views, FLAGS.height, FLAGS.width, 3],
                       name='inputs')
    mid_level_X = tf.placeholder(tf.float32,
                       [FLAGS.num_views, None, 35, 35, 384], # inputs of 4 x Inception-A blocks
                       name='inputs')
    is_training = tf.placeholder(tf.bool)
    dropout_keep_prob = tf.placeholder(tf.float32)
    grouping_scheme = tf.placeholder(tf.bool, [NUM_GROUP, FLAGS.num_views])
    grouping_weight = tf.placeholder(tf.float32, [NUM_GROUP, 1])

    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        # grouping module
        d_scores, raw_desc = gvcnn.discrimination_score(X)

    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        # GVCNN
        logits, _, end_points = gvcnn.gvcnn(mid_level_X,
                                            grouping_scheme,
                                            grouping_weight,
                                            num_classes,
                                            is_training,
                                            dropout_keep_prob)

    prediction = tf.nn.softmax(logits)
    predicted_labels = tf.argmax(prediction, 1)

    # prediction = tf.argmax(logits, 1, name='prediction')
    # correct_prediction = tf.equal(prediction, ground_truth)
    # confusion_matrix = tf.confusion_matrix(
    #     ground_truth, prediction, num_classes=num_classes)
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    ################
    # Prepare data
    ################
    filenames = tf.placeholder(tf.string, shape=[])
    eval_dataset = data.Dataset(filenames, FLAGS.height, FLAGS.width)
    iterator = eval_dataset.dataset.make_initializable_iterator()
    next_batch = iterator.get_next()

    sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())

        # Create a saver object which will save all the variables
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=1.0)
        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path
        saver.restore(sess, checkpoint_path)

        global_step = checkpoint_path.split('/')[-1].split('-')[-1]

        # Get the number of training/validation steps per epoch
        batches = int(MODELNET_EVAL_DATA_SIZE / FLAGS.batch_size)
        if MODELNET_EVAL_DATA_SIZE % FLAGS.batch_size > 0:
            batches += 1

        ##################################################
        # prediction & make results into csv file.
        ##################################################
        start_time = datetime.datetime.now()
        print("Start prediction: {}".format(start_time))

        id2name = {i: name for i, name in enumerate(labels)}
        submission = {}

        eval_filenames = os.path.join(FLAGS.dataset_path)
        sess.run(iterator.initializer, feed_dict={filenames: eval_filenames})
        count = 0;
        for i in range(batches):
            batch_xs, filename = sess.run(next_batch)
            # # Verify image
            # n_batch = batch_xs.shape[0]
            # for i in range(n_batch):
            #     img = batch_xs[i]
            #     # scipy.misc.toimage(img).show()
            #     # Or
            #     img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            #     cv2.imwrite('/home/ace19/Pictures/' + str(i) + '.png', img)
            #     # cv2.imshow(str(fnames), img)
            #     cv2.waitKey(100)
            #     cv2.destroyAllWindows()

            # Sets up a graph with feeds and fetches for partial run.
            handle = sess.partial_run_setup([d_scores, raw_desc, predicted_labels],
                                            [X, mid_level_X, grouping_scheme,
                                             grouping_weight, is_training, dropout_keep_prob])

            scores, r_desc = sess.partial_run(handle,
                                              [d_scores, raw_desc],
                                              feed_dict={X: batch_xs})
            schemes = gvcnn.grouping_scheme(scores, NUM_GROUP, FLAGS.num_views)
            weights = gvcnn.grouping_weight(scores, schemes)

            pred = sess.partial_run(handle,
                                    predicted_labels,
                                    feed_dict={mid_level_X: r_desc,
                                               grouping_scheme: schemes,
                                               grouping_weight: weights,
                                               is_training: False,
                                               dropout_keep_prob: 1.0})
            size = len(filename)
            for n in range(size):
                submission[filename[n].decode('UTF-8')] = id2name[pred[n]]

            count += size
            tf.logging.info('Total count: #%d' % count)

        end_time = datetime.datetime.now()
        tf.logging.info('#%d Data, End prediction: %s' % (MODELNET_EVAL_DATA_SIZE, end_time))
        tf.logging.info('prediction waste time: %s' % (end_time - start_time))

    ######################################
    # make submission.csv for kaggle
    ######################################
    if not os.path.exists(FLAGS.result_dir):
        os.makedirs(FLAGS.result_dir)

    fout = open(
        os.path.join(FLAGS.result_dir,
                     FLAGS.model_architecture + '-#' +
                     global_step + '.csv'),
        'w', encoding='utf-8', newline='')
    writer = csv.writer(fout)
    writer.writerow(['image', 'label'])
    for key in sorted(submission.keys()):
        writer.writerow([key, submission[key]])
    fout.close()

if __name__ == '__main__':
    tf.app.run()
