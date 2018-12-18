import time
import numpy as np

import tensorflow as tf
import gvcnn
from utils import train_utils

from slim.deployment import model_deploy

slim = tf.contrib.slim

prefetch_queue = slim.prefetch_queue

flags = tf.app.flags

FLAGS = flags.FLAGS



LABELS_CLASS = 'labels_class'
IMAGE = 'image'
HEIGHT = 'height'
WIDTH = 'width'
IMAGE_NAME = 'image_name'
LABEL = 'label'
NUM_VIEWS = 12
NUM_GROUP = 5


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


flags.DEFINE_integer('train_batch_size', 8,
                     'The value of the weight decay for training.')


# Settings for fine-tuning the network.
flags.DEFINE_string('tf_initial_checkpoint', './checkpoint',
                    'The initial checkpoint in tensorflow format.')


# Dataset settings.
flags.DEFINE_string('dataset', 'pascal_voc_seg',
                    'Name of the segmentation dataset.')

flags.DEFINE_string('train_split', 'train',
                    'Which split of the dataset to be used for training')

flags.DEFINE_string('dataset_dir', './datasets/pascal_voc_seg/tfrecord',
                    'Where the dataset reside.')


flags.DEFINE_integer('num_views', 4, 'Number of views')



def test():
    train_batch_size = 1
    num_views = 8       # set of views
    # eval_batch_size = 2
    height, width = 224, 224
    num_classes = 3
    num_group = 5

    # TODO: Use Princeton ModelNet dataset
    # TODO: TFRecord
    train_inputs = tf.placeholder(tf.float32, [None, num_views, height, width, 3])
    dropout_keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)

    # Make grouping module
    d_scores, g_scheme, g_weight = \
        gvcnn.make_grouping_module(train_inputs,
                                   num_group,
                                   is_training,
                                   dropout_keep_prob=dropout_keep_prob)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        inputs = tf.random_uniform((train_batch_size, num_views, height, width, 3))
        scores, scheme, weight = sess.run([d_scores, g_scheme, g_weight],
                                          feed_dict={train_inputs: inputs.eval(),
                                                     dropout_keep_prob: 0.8,
                                                     is_training: True})
        sess.close()

    group_scheme = train_utils.refine_dict(scheme)
    logits, end_points = gvcnn.gvcnn(train_inputs,
                                     group_scheme,
                                     is_training,
                                     dropout_keep_prob=dropout_keep_prob)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        sess.run([logits], feed_dict={train_inputs: inputs.eval(),
                                      dropout_keep_prob: 0.8,
                                      is_training: True})

    tf.logging.info("++++++")


def read_lists(list_of_lists_file):
    listfile_labels = np.loadtxt(list_of_lists_file, dtype=str).tolist()
    listfiles, labels  = zip(*[(l[0], int(l[1])) for l in listfile_labels])
    return listfiles, labels


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    test()

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
