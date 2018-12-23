import os

import tensorflow as tf

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('target_dir', '/home/ace19/dl_data/modelnet/cone/test',
                    'target_dir')

flags.DEFINE_string('source', '/home/ace19/dl_data/ModelNet40/cone/test',
                    'source')


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    off_files = os.listdir(FLAGS.source)
    off_files.sort()

    for f in off_files:
        if '.off' in f:
            path = os.path.join(FLAGS.target_dir, f)
            os.makedirs(path)



if __name__ == '__main__':
    tf.app.run()