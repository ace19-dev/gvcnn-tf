import tensorflow as tf

import os

import data
import gvcnn
from utils import train_utils

from slim.nets import inception_v2

slim = tf.contrib.slim

flags = tf.app.flags

FLAGS = flags.FLAGS


IMAGE = 'image'
LABEL = 'label'
OUTPUT_TYPE = 'classification'


# Settings for logging.
flags.DEFINE_string('train_logdir', './checkpoint',
                    'Where the checkpoint and logs are stored.')
flags.DEFINE_integer('log_steps', 10,
                     'Display logging information at every log_steps.')
flags.DEFINE_integer('save_interval_secs', 1200,
                     'How often, in seconds, we save the model to disk.')
flags.DEFINE_boolean('save_summaries_images', False,
                     'Save sample inputs, labels, and semantic predictions as '
                     'images to summary.')
flags.DEFINE_string('summaries_dir', './checkpoint/models/train_logs',
                     'Where to save summary logs for TensorBoard.')

flags.DEFINE_enum('learning_policy', 'step', ['step'],
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

flags.DEFINE_float('last_layer_gradient_multiplier', 1.0,
                   'The gradient multiplier for last layers, which is used to '
                   'boost the gradient of last layers if the value > 1.')

# Settings for fine-tuning the network.
flags.DEFINE_string('tf_initial_checkpoint', None,    # ./checkpoint/mobile.ckpt-20
                    'The initial checkpoint in tensorflow format.')
# Set to False if one does not want to re-use the trained classifier weights.
flags.DEFINE_boolean('initialize_last_layer', True,
                     'Initialize the last layer.')
flags.DEFINE_boolean('last_layers_contain_logits_only', False,
                     'Only consider logits as last layers or not.')
flags.DEFINE_integer('slow_start_step', 0,
                     'Training model with small learning rate for few steps.')
flags.DEFINE_float('slow_start_learning_rate', 1e-4,
                   'Learning rate employed during slow start.')

# Dataset settings.
flags.DEFINE_string('dataset_dir', '/home/ace19/dl_data/modelnet',
                    'Where the dataset reside.')

flags.DEFINE_integer('how_many_training_epochs', 3, 'How many training loops to run')
flags.DEFINE_integer('batch_size', 16, 'batch size')
flags.DEFINE_integer('num_views', 8, 'number of views')
flags.DEFINE_string('height_weight', '224,224', 'height and weight')
flags.DEFINE_integer('num_classes', 7, 'number of classes')
flags.DEFINE_integer('num_group', 5, 'number of grouping')


SCOPE = "GoogLeNet"

def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    h_w = list(map(int, FLAGS.height_weight.split(',')))
    # test.test(h_w,
    #           FLAGS.num_views,
    #           FLAGS.num_group,
    #           FLAGS.num_classes,
    #           FLAGS.batch_size)
    # test2()
    # test3()
    # test4()

    tf.gfile.MakeDirs(FLAGS.train_logdir)
    tf.logging.info('Creating train logdir: %s', FLAGS.train_logdir)

    with tf.Graph().as_default() as graph:
        dataset = data.Data(FLAGS.dataset_dir, h_w)

        global_step = tf.train.get_or_create_global_step()

        # Define the model
        x = tf.placeholder(tf.float32, [None, FLAGS.num_views, h_w[0], h_w[1], 3], name=IMAGE)
        ground_truth = tf.placeholder(tf.int64, [None], name=LABEL)
        is_training = tf.placeholder(tf.bool)
        dropout_keep_prob = tf.placeholder(tf.float32)
        grouping_scheme = tf.placeholder(tf.bool, [FLAGS.num_group, FLAGS.num_views])
        grouping_weight = tf.placeholder(tf.float32, [FLAGS.num_group, 1])

        # TODO: use each graphs for grouping module and GVCNN ??
        # grouping module
        d_scores, g_scheme = gvcnn.grouping_module(x,
                                                   FLAGS.num_group,
                                                   is_training,
                                                   dropout_keep_prob=dropout_keep_prob)
        # GVCNN
        logits = gvcnn.gvcnn(x,
                             grouping_scheme,
                             grouping_weight,
                             FLAGS.num_classes,
                             is_training,
                             scope=SCOPE,
                             dropout_keep_prob=dropout_keep_prob)

        # make a trainable variable not trainable
        train_utils.edit_trainable_variables('FCN')

        # Define loss
        tf.losses.sparse_softmax_cross_entropy(labels=ground_truth, logits=logits)

        # Gather update_ops. These contain, for example,
        # the updates for the batch_norm variables created by model.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, SCOPE)
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # Add summaries for model variables.
        for model_var in slim.get_model_variables():
            summaries.add(tf.summary.histogram(model_var.op.name, model_var))

        # TODO: need to be edited
        # Add summaries for images, labels, predictions
        # if FLAGS.save_summaries_images:
        #     summary_image = graph.get_tensor_by_name('%s:0' % (IMAGE))
        #     summaries.add(
        #         tf.summary.image('samples/%s' % IMAGE, summary_image))
        #
        #     label = graph.get_tensor_by_name('%s:0' % (LABEL))
        #     # Scale up summary image pixel values for better visualization.
        #     pixel_scaling = max(1, 255 // FLAGS.num_classes)
        #     summary_label = tf.cast(label * pixel_scaling, tf.uint8)
        #     summaries.add(
        #         tf.summary.image('samples/%s' % LABEL, summary_label))
        #     output = graph.get_tensor_by_name(
        #         ('%s/%s:0' % (OUTPUT_TYPE)).strip('/'))
        #     predictions = tf.expand_dims(tf.argmax(output, 3), -1)
        #
        #     summary_predictions = tf.cast(predictions * pixel_scaling, tf.uint8)
        #     summaries.add(
        #         tf.summary.image(
        #             'samples/%s' % OUTPUT_TYPE, summary_predictions))

        # Add summaries for losses.
        for loss in tf.get_collection(tf.GraphKeys.LOSSES, SCOPE):
        # for loss in tf.get_collection(tf.GraphKeys.LOSSES):
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

        total_loss, grads_and_vars = train_utils.optimize(optimizer, scope=SCOPE)
        # total_loss, grads_and_vars = train_utils.optimize(optimizer)
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
            train_op = tf.identity(total_loss, name='train_op')

        prediction = tf.argmax(logits, 1, name='prediction')
        correct_prediction = tf.equal(prediction, ground_truth)
        confusion_matrix = tf.confusion_matrix(
            ground_truth, prediction, num_classes=FLAGS.num_classes)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        summaries.add(tf.summary.scalar('accuracy', accuracy))

        # Add the summaries. These contain the summaries
        # created by model and either optimize() or _gather_loss().
        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES, SCOPE))
        # summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries))
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir, graph)

        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())

            # Create a saver object which will save all the variables
            # TODO:
            saver = tf.train.Saver()
            if FLAGS.tf_initial_checkpoint:
                saver.restore(sess, FLAGS.tf_initial_checkpoint)
            # saver = tf.train.Saver(
            #     keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)

            start_epoch = 0
            # Get the number of training/validation steps per epoch
            t_batches = int(dataset.data_size() / FLAGS.batch_size)
            if dataset.data_size() % FLAGS.batch_size > 0:
                t_batches += 1
            # v_batches = int(dataset.data_size() / FLAGS.batch_size)
            # if val_data.data_size() % FLAGS.batch_size > 0:
            #     v_batches += 1

            ############################
            # Training loop.
            ############################
            for training_epoch in range(start_epoch, FLAGS.how_many_training_epochs):
                print("------------------------")
                print(" Epoch {} ".format(training_epoch + 1))
                print("------------------------")

                dataset.shuffle()
                for step in range(t_batches):
                    # Pull the image batch we'll use for training.
                    train_batch_xs, train_batch_ys = dataset.next_batch(step, FLAGS.batch_size)

                    # For debugging
                    # img = train_batch_xs[0]
                    # plt.hist(img.ravel())
                    # plt.show()

                    scores, scheme = \
                        sess.run([d_scores, g_scheme], feed_dict={x: train_batch_xs.eval(),
                                                                  is_training: True,
                                                                  dropout_keep_prob: 0.8})
                    s = gvcnn.refine_group(scheme, FLAGS.num_group, FLAGS.num_views)
                    w = gvcnn.group_weight(scores, s)

                    # Run the graph with this batch of training data.
                    lr, train_summary, train_accuracy, train_loss, _ = \
                        sess.run([learning_rate, summary_op, accuracy, total_loss, train_op],
                                 feed_dict={x: train_batch_xs.eval(),
                                            ground_truth: train_batch_ys,
                                            grouping_scheme: s,
                                            grouping_weight: w,
                                            is_training: True,
                                            dropout_keep_prob: 0.8})

                    train_writer.add_summary(train_summary)
                    tf.logging.info('Epoch #%d, Step #%d, rate %f, accuracy %.1f%%, loss %f' %
                                    (training_epoch, step, lr, train_accuracy * 100, train_loss))


                ###################################################
                # TODO: Validate the model on the validation set
                ###################################################

                # Save the model checkpoint periodically.
                if (training_epoch <= FLAGS.how_many_training_epochs-1):
                    checkpoint_path = os.path.join(FLAGS.train_logdir, 'GVCNN.ckpt')
                    tf.logging.info('Saving to "%s-%d"', checkpoint_path, training_epoch)
                    saver.save(sess, checkpoint_path, global_step=global_step)


if __name__ == '__main__':
    tf.app.run()
