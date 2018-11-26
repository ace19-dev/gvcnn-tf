import tensorflow as tf

from slim.deployment import model_deploy

flags = tf.app.flags

FLAGS = flags.FLAGS


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
flags.DEFINE_integer('save_summaries_images', True,
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


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    # Set up deployment (i.e. multi-GPUs and/or multi-replicas).
    config = model_deploy.DeploymentConfig(
        num_clones=FLAGS.num_clones,
        clone_on_cpu=FLAGS.clones_on_cpu,
        replica_id=FLAGS.task,
        num_replicas=FLAGS.num_replicas,
        num_ps_tasks=FLAGS.num_ps_tasks)

    # Split the batch across GPUs.
    assert FLAGS.train_batch_size % config.num_clones == 0, (
        'Training batch size not divisible by number of clones (GPUs).')

    clone_batch_size = FLAGS.train_batch_size // config.num_clones

    # Get dataset infomation


    tf.gfile.MakeDirs(FLAGS.train_logdir)
    tf.logging.info('Training on %s set', FLAGS.train_split)






























if __name__ == '__main__':
    tf.app.run()























