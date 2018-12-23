import time
import numpy as np
import cv2

from six.moves import xrange

import tensorflow as tf
import gvcnn
import model
import data

from nets import googLeNet


slim = tf.contrib.slim

# prefetch_queue = slim.prefetch_queue

flags = tf.app.flags

FLAGS = flags.FLAGS



LABELS_CLASS = 'labels_class'
IMAGE = 'image'
HEIGHT = 'height'
WIDTH = 'width'
IMAGE_NAME = 'image_name'
LABEL = 'label'


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


flags.DEFINE_integer('how_many_training_epochs', 10, 'How many training loops to run')

flags.DEFINE_integer('train_batch_size', 16, 'batch size')
flags.DEFINE_integer('num_views', 8, 'number of views')
flags.DEFINE_string('height_weight', '224,224', 'height and weight')
flags.DEFINE_integer('num_classes', 7, 'number of classes')



def test():
    h_w = list(map(int, FLAGS.height_weight.split(',')))

    # TODO: remove comment soon
    # dataset = data.Data(FLAGS.dataset_dir, h_w)
    # images, labels = d.next_batch(0, FLAGS.train_batch_size)

    x = tf.placeholder(tf.float32, [None, FLAGS.num_views, h_w[0], h_w[1], 3])
    gt = tf.placeholder(tf.int32, [None])
    is_training = tf.placeholder(tf.bool)
    dropout_keep_prob = tf.placeholder(tf.float32)

    view_discrimination_scores = tf.placeholder(tf.float32, [FLAGS.num_views])
    group_scheme = tf.placeholder(tf.int32, [FLAGS.num_views])

        # # Create a saver object which will save all the variables
        # saver = tf.train.Saver()
        #
        # start_epoch = 1
        # start_checkpoint_epoch = 0
        # if FLAGS.start_checkpoint:
        #     # model.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
        #     saver.restore(sess, FLAGS.start_checkpoint)
        #     tmp = FLAGS.start_checkpoint
        #     tmp = tmp.split('-')
        #     tmp.reverse()
        #     start_checkpoint_epoch = int(tmp[0])
        #     start_epoch = start_checkpoint_epoch + 1
        #
        # ############################
        # # Training loop.
        # ############################
        # for training_epoch in xrange(start_epoch, training_epochs_max + 1):

    # Make grouping module
    d_scores, g_scheme = \
        gvcnn.make_grouping_module(x,
                                   is_training,
                                   dropout_keep_prob=dropout_keep_prob)

    predictions = gvcnn.gvcnn(x,
                              view_discrimination_scores,
                              group_scheme,
                              FLAGS.num_classes,
                              is_training,
                              dropout_keep_prob=dropout_keep_prob)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        inputs = tf.random_uniform((FLAGS.train_batch_size, FLAGS.num_views, h_w[0], h_w[1], 3))
        scores, scheme = \
            sess.run([d_scores, g_scheme], feed_dict={x: inputs.eval(),
                                                      is_training: True,
                                                      dropout_keep_prob: 0.8})

        # group_scheme = gvcnn.refine_scheme(scheme)
        # group_weight = gvcnn.group_weight(scores, group_scheme)
    # predictions = gvcnn.gvcnn(x,
    #                           group_scheme,
    #                           group_weight,
    #                           FLAGS.num_classes,
    #                           is_training,
    #                           dropout_keep_prob=dropout_keep_prob)

        pred = sess.run([predictions], feed_dict={x: inputs.eval(),
                                                  view_discrimination_scores: scores,
                                                  group_scheme: scheme,
                                                  dropout_keep_prob: 0.8,
                                                  is_training: True})

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

    model.inference_multiview(inputs, 10, 0.8)


def read_lists(list_of_lists_file):
    listfile_labels = np.loadtxt(list_of_lists_file, dtype=str).tolist()
    listfiles, labels  = zip(*[(l[0], int(l[1])) for l in listfile_labels])
    return listfiles, labels


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    test()
    # test2()
    # test3()


    # # Set up deployment (i.e. multi-GPUs and/or multi-replicas).
    # config = model_deploy.DeploymentConfig(
    #     num_clones=FLAGS.num_clones,
    #     clone_on_cpu=FLAGS.clones_on_cpu,
    #     replica_id=FLAGS.task,
    #     num_replicas=FLAGS.num_replicas,
    #     num_ps_tasks=FLAGS.num_ps_tasks)
    #
    # # Split the batch across GPUs.
    # assert FLAGS.train_batch_size % config.num_clones == 0, (
    #     'Training batch size not divisible by number of clones (GPUs).')
    #
    # clone_batch_size = FLAGS.train_batch_size // config.num_clones
    #
    # # Get dataset infomation
    # st = time.time()
    # print('start loading data')
    #
    # # listfiles_train, labels_train = read_lists(g_.TRAIN_LOL)
    # # listfiles_val, labels_val = read_lists(g_.VAL_LOL)
    # # dataset_train = Dataset(listfiles_train, labels_train, subtract_mean=False, V=g_.NUM_VIEWS)
    # # dataset_val = Dataset(listfiles_val, labels_val, subtract_mean=False, V=g_.NUM_VIEWS)
    #
    # print('done loading data, time=', time.time() - st)
    #
    #
    # tf.gfile.MakeDirs(FLAGS.train_logdir)
    # tf.logging.info('Training on %s set', FLAGS.train_split)
    #
    # with tf.Graph().as_default() as graph:
    #     # placeholders for graph input
    #     view_ = tf.placeholder('float32', shape=(None, FLAGS.num_views, 224, 224, 3), name='views')
    #     y_ = tf.placeholder('int64', shape=(None), name='y')
    #     keep_prob_ = tf.placeholder('float32')

    # with tf.Graph().as_default() as graph:
    #     with tf.device(config.inputs_device()):
    #         samples = input_generator.get(
    #             dataset,
    #             FLAGS.train_crop_size,
    #             clone_batch_size,
    #             min_resize_value=FLAGS.min_resize_value,
    #             max_resize_value=FLAGS.max_resize_value,
    #             resize_factor=FLAGS.resize_factor,
    #             min_scale_factor=FLAGS.min_scale_factor,
    #             max_scale_factor=FLAGS.max_scale_factor,
    #             scale_factor_step_size=FLAGS.scale_factor_step_size,
    #             dataset_split=FLAGS.train_split,
    #             is_training=True,
    #             model_variant=FLAGS.model_variant)
    #         inputs_queue = prefetch_queue.prefetch_queue(
    #             samples, capacity=128 * config.num_clones)
    #
    #     # Create the global step on the device storing the variables.
    #     with tf.device(config.variables_device()):
    #         global_step = tf.train.get_or_create_global_step()
    #
    #         # graph outputs
    #         # fc8 = model.inference_multiview(view_, g_.NUM_CLASSES, keep_prob_)
    #         # loss = model.loss(fc8, y_)
    #         # train_op = model.train(loss, global_step, data_size)
    #         # prediction = model.classify(fc8)
    #
    #         # Define the model and create clones.
    #         model_fn = model.FCN
    #         model_args = (inputs_queue, {
    #             common.OUTPUT_TYPE: dataset.num_classes
    #         }, dataset.ignore_label)
    #         clones = model_deploy.create_clones(config, model_fn, args=model_args)
    #
    #         # Gather update_ops from the first clone. These contain, for example,
    #         # the updates for the batch_norm variables created by model_fn.
    #         first_clone_scope = config.clone_scope(0)
    #         update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)
    #
    #     # Gather initial summaries.
    #     summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    #
    #     # Add summaries for model variables.
    #     for model_var in slim.get_model_variables():
    #         summaries.add(tf.summary.histogram(model_var.op.name, model_var))
    #
    #     # Add summaries for images, labels, semantic predictions
    #     if FLAGS.save_summaries_images:
    #         summary_image = graph.get_tensor_by_name(
    #             ('%s/%s:0' % (first_clone_scope, IMAGE)).strip('/'))
    #         summaries.add(
    #             tf.summary.image('samples/%s' % IMAGE, summary_image))
    #
    #         first_clone_label = graph.get_tensor_by_name(
    #             ('%s/%s:0' % (first_clone_scope, LABEL)).strip('/'))
    #         # Scale up summary image pixel values for better visualization.
    #         pixel_scaling = max(1, 255 // dataset.num_classes)
    #         summary_label = tf.cast(first_clone_label * pixel_scaling, tf.uint8)
    #         summaries.add(
    #             tf.summary.image('samples/%s' % LABEL, summary_label))
    #
    #         first_clone_output = graph.get_tensor_by_name(
    #             ('%s/%s:0' % (first_clone_scope, common.OUTPUT_TYPE)).strip('/'))
    #         predictions = tf.expand_dims(tf.argmax(first_clone_output, 3), -1)
    #
    #         summary_predictions = tf.cast(predictions * pixel_scaling, tf.uint8)
    #         summaries.add(
    #             tf.summary.image(
    #                 'samples/%s' % common.OUTPUT_TYPE, summary_predictions))
    #
    #     # Add summaries for losses.
    #     for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
    #         summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))
    #
    #     # Build the optimizer based on the device specification.
    #     with tf.device(config.optimizer_device()):
    #         learning_rate = train_utils.get_model_learning_rate(
    #             FLAGS.learning_policy, FLAGS.base_learning_rate,
    #             FLAGS.learning_rate_decay_step, FLAGS.learning_rate_decay_factor,
    #             FLAGS.training_number_of_steps, FLAGS.learning_power,
    #             FLAGS.slow_start_step, FLAGS.slow_start_learning_rate)
    #         optimizer = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)
    #         summaries.add(tf.summary.scalar('learning_rate', learning_rate))
    #
    #     startup_delay_steps = FLAGS.task * FLAGS.startup_delay_steps
    #     for variable in slim.get_model_variables():
    #         summaries.add(tf.summary.histogram(variable.op.name, variable))
    #
    #     with tf.device(config.variables_device()):
    #         total_loss, grads_and_vars = model_deploy.optimize_clones(
    #             clones, optimizer)
    #         total_loss = tf.check_numerics(total_loss, 'Loss is inf or nan.')
    #         summaries.add(tf.summary.scalar('total_loss', total_loss))
    #
    #         # Modify the gradients for biases and last layer variables.
    #         last_layers = model.get_extra_layer_scopes(
    #             FLAGS.last_layers_contain_logits_only)
    #         grad_mult = train_utils.get_model_gradient_multipliers(
    #             last_layers, FLAGS.last_layer_gradient_multiplier)
    #         if grad_mult:
    #             grads_and_vars = slim.learning.multiply_gradients(
    #                 grads_and_vars, grad_mult)
    #
    #         # Create gradient update op.
    #         grad_updates = optimizer.apply_gradients(
    #             grads_and_vars, global_step=global_step)
    #         update_ops.append(grad_updates)
    #         update_op = tf.group(*update_ops)
    #         with tf.control_dependencies([update_op]):
    #             train_tensor = tf.identity(total_loss, name='train_op')
    #
    #     # Add the summaries from the first clone. These contain the summaries
    #     # created by model_fn and either optimize_clones() or _gather_clone_loss().
    #     summaries |= set(
    #         tf.get_collection(tf.GraphKeys.SUMMARIES, first_clone_scope))
    #
    #     # Merge all summaries together.
    #     summary_op = tf.summary.merge(list(summaries))
    #
    #     # Soft placement allows placing on CPU ops without GPU implementation.
    #     session_config = tf.ConfigProto(
    #         allow_soft_placement=True, log_device_placement=False)
    #
    #     # Start the training.
    #     slim.learning.train(
    #         train_tensor,
    #         logdir=FLAGS.train_logdir,
    #         log_every_n_steps=FLAGS.log_steps,
    #         master=FLAGS.master,
    #         number_of_steps=FLAGS.training_number_of_steps,
    #         is_chief=(FLAGS.task == 0),
    #         session_config=session_config,
    #         startup_delay_steps=startup_delay_steps,
    #         init_fn=train_utils.get_model_init_fn(
    #             FLAGS.train_logdir,
    #             FLAGS.tf_initial_checkpoint,
    #             FLAGS.initialize_last_layer,
    #             last_layers,
    #             ignore_missing_vars=True),
    #         summary_op=summary_op,
    #         save_summaries_secs=FLAGS.save_summaries_secs,
    #         save_interval_secs=FLAGS.save_interval_secs)



if __name__ == '__main__':
    tf.app.run()
