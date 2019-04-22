import tensorflow as tf

import os

flags = tf.app.flags

FLAGS = flags.FLAGS


flags.DEFINE_string('source_dir', '/home/ace19/dl_data/ModelNet10',
                    'source dir')

flags.DEFINE_string('target_dir', '/home/ace19/dl_data/ModelNet10',
                    'target dir')

flags.DEFINE_string('target_file_ext', '.obj',
                    'target file extension')

flags.DEFINE_string('dataset_category', 'train',
                    'train or test')


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    root = os.listdir(FLAGS.source_dir)
    root.sort()

    for cls in root:
        if not os.path.isdir(os.path.join(FLAGS.source_dir, cls)):
            continue

        dataset = os.path.join(FLAGS.source_dir, cls, FLAGS.dataset_category)
        off_files = os.listdir(dataset)
        off_files.sort()

        total = len(off_files)
        for i, file in enumerate(off_files):
            if i % 50 == 0:
                tf.logging.info('\n\ncompleted \'%s\': %d/%d' % (cls, i, total))

            file_name = os.path.basename(file)[:-4]
            output_file_path = os.path.join(FLAGS.target_dir, cls, FLAGS.dataset_category, file_name + FLAGS.target_file_ext)
            cmd = 'ctmconv ' + os.path.join(dataset, file) + ' ' + output_file_path
            os.system(cmd)


if __name__ == '__main__':
    tf.app.run()