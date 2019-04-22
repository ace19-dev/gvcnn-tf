import os

import tensorflow as tf

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('source_dir', '/home/ace19/dl_data/ModelNet10',
                    'source dir')

flags.DEFINE_string('target_dir', '/home/ace19/dl_data/modelnet',
                    'target dir')

flags.DEFINE_string('dataset_category', 'train',
                    'train or test')


def main(unused_argv):
    root = os.listdir(FLAGS.source_dir)
    root.sort()

    for cls in root:
        if not os.path.isdir(os.path.join(FLAGS.source_dir, cls)):
            continue

        dataset = os.path.join(FLAGS.source_dir, cls, FLAGS.dataset_category)
        data_list = os.listdir(dataset)

        for f in data_list:
            if '.off' in f:
                target_dir = os.path.join(FLAGS.target_dir, cls, FLAGS.dataset_category)
                p = os.path.join(target_dir, f)
                os.makedirs(p)



if __name__ == '__main__':
    tf.app.run()