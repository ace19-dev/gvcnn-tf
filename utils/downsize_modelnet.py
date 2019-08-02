from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

import tensorflow as tf


flags = tf.app.flags
flags.DEFINE_string('dataset_dir',
                    '/home/ace19/dl_data/modelnet12/view/classes',
                    'Root Directory to modelnet12 dataset.')
# flags.DEFINE_string('output_path',
#                     '/home/ace19/dl_data/modelnet10_sv',
#                     'Path to output')
flags.DEFINE_string('dataset_category',
                    'test',
                    'dataset category, train|validate|test')

FLAGS = flags.FLAGS


def main(_):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    cls_lst = os.listdir(FLAGS.dataset_dir)
    for i, label in enumerate(cls_lst):
        category_path = os.path.join(FLAGS.dataset_dir, label, FLAGS.dataset_category)
        off_lst = os.listdir(category_path)
        for off_file in off_lst:
            off_path = os.path.join(category_path, off_file)
            img_lst = os.listdir(off_path)
            for img in img_lst:
                img_path = os.path.join(off_path, img)
                if os.path.isfile(img_path):
                   if not int(img.split('.')[1]) % 2 == 0:
                       os.remove(img_path)

if __name__ == '__main__':
    tf.compat.v1.app.run()