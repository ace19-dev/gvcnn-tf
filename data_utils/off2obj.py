import tensorflow as tf

import os
import subprocess

flags = tf.app.flags

FLAGS = flags.FLAGS


flags.DEFINE_string('input_dir', '/home/ace19/dl_data/ModelNet40/xbox',
                    'input dir')

flags.DEFINE_string('output_dir', '/home/ace19/dl_data/ModelNet40/xbox',
                    'output dir')

flags.DEFINE_string('output_file_ext', '.obj',
                    'output file extension')

flags.DEFINE_string('sub_dir', 'test',
                    'train or test')


def main(unused_argv):
    input = os.path.join(FLAGS.input_dir, FLAGS.sub_dir)
    off_files = os.listdir(input)
    off_files.sort()

    for file in off_files:
        file_name = os.path.basename(file)[:-4]
        output_file_path = os.path.join(FLAGS.output_dir, FLAGS.sub_dir, file_name + FLAGS.output_file_ext)
        cmd = 'ctmconv ' + \
              os.path.join(input, file) + ' ' \
              + output_file_path
        os.system(cmd)
        # subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    tf.app.run()