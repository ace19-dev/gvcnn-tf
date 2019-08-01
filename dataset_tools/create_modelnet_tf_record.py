"""
Convert PatchCamelyon (PCam) dataset to TFRecord for classification.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os
import random
import numpy as np

import PIL.Image
import tensorflow as tf

from dataset_tools import dataset_util


# RANDOM_SEED = 8045

flags = tf.app.flags
flags.DEFINE_string('dataset_dir',
                    '/home/ace19/dl_data/modelnet40/view/classes',
                    'Root Directory to raw modelnet dataset.')
flags.DEFINE_string('output_dir',
                    '/home/ace19/dl_data/modelnet40/tfrecords',
                    'Path to output TFRecord')
flags.DEFINE_string('dataset_category',
                    'test',
                    'dataset category, train|validate|test')

FLAGS = flags.FLAGS

_FILE_PATTERN = 'modelnet40_%s_%s.tfrecord'


def get_data_map_dict(label_to_index):
    label_map_dict = {}
    view_map_dict = {}

    cls_lst = os.listdir(FLAGS.dataset_dir)
    for i, cls in enumerate(cls_lst):
        if not os.path.isdir(os.path.join(FLAGS.dataset_dir, cls)):
            continue

        data_path = os.path.join(FLAGS.dataset_dir, cls, FLAGS.dataset_category)
        img_lst = os.listdir(data_path)
        for n, img in enumerate(img_lst):
            img_path = os.path.join(data_path, img)
            view_lst = os.listdir(img_path)
            views = []
            for k, view in enumerate(view_lst):
                v_path = os.path.join(img_path, view)
                views.append(v_path)
            label_map_dict[img] = label_to_index[cls]
            view_map_dict[img] = views

    return label_map_dict, view_map_dict


def dict_to_tf_example(image,
                       label_map_dict=None,
                       view_map_dict=None):
    """
    Args:
      image: a single image name
      label_map_dict: A map from string label names to integers ids.
      image_subdirectory: String specifying subdirectory within the
        PCam dataset directory holding the actual image data.

    Returns:
      example: The converted tf.Example.

    Raises:
      ValueError: if the image pointed to by image is not a valid PNG
    """
    # full_path = os.path.join(dataset_directory, image_subdirectory, image_name)

    filenames = []
    sourceids = []
    encoded_pngs = []
    widths = []
    heights = []
    formats = []
    labels = []
    keys = []

    view_lst = view_map_dict[image]
    label = label_map_dict[image]
    for i, view_path in enumerate(view_lst):
        filenames.append(view_path.encode('utf8'))
        sourceids.append(view_path.encode('utf8'))
        with tf.gfile.GFile(view_path, 'rb') as fid:
            encoded_png = fid.read()
            encoded_pngs.append(encoded_png)
        encoded_png_io = io.BytesIO(encoded_png)
        image = PIL.Image.open(encoded_png_io)
        width, height = image.size
        widths.append(width)
        heights.append(height)

        format = image.format
        formats.append(format.encode('utf8'))
        if format!= 'PNG':
            raise ValueError('Image format not PNG')
        key = hashlib.sha256(encoded_png).hexdigest()
        keys.append(key.encode('utf8'))
        # labels.append(label)

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_list_feature(heights),
        'image/width': dataset_util.int64_list_feature(widths),
        'image/filename': dataset_util.bytes_list_feature(filenames),
        'image/source_id': dataset_util.bytes_list_feature(sourceids),
        'image/key/sha256': dataset_util.bytes_list_feature(keys),
        'image/encoded': dataset_util.bytes_list_feature(encoded_pngs),
        'image/format': dataset_util.bytes_list_feature(formats),
        'image/label': dataset_util.int64_feature(label),
        # 'image/text': dataset_util.bytes_feature('label_text'.encode('utf8'))
    }))
    return example


def main(_):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    options = tf.io.TFRecordOptions(tf.compat.v1.python_io.TFRecordCompressionType.GZIP)
    # writer = tf.python_io.TFRecordWriter(FLAGS.output_dir, options=options)

    dataset_lst = os.listdir(FLAGS.dataset_dir)
    dataset_lst.sort()
    label_to_index = {}
    for i, cls in enumerate(dataset_lst):
        cls_path = os.path.join(FLAGS.dataset_dir, cls)
        if os.path.isdir(cls_path):
            label_to_index[cls] = i

    label_map_dict, view_map_dict = get_data_map_dict(label_to_index)

    tf.compat.v1.logging.info('Reading from modelnet dataset.')
    cls_lst = os.listdir(FLAGS.dataset_dir)
    for i, label in enumerate(cls_lst):
        tfrecord_name = os.path.join(FLAGS.output_dir,
                                     _FILE_PATTERN % (FLAGS.dataset_category, label))
        tf.compat.v1.logging.info('tfrecord name %s: ', tfrecord_name)
        writer = tf.io.TFRecordWriter(tfrecord_name, options=options)

        data_path = os.path.join(FLAGS.dataset_dir, label, FLAGS.dataset_category)
        if not os.path.isdir(data_path):
            continue
        img_lst = os.listdir(data_path)
        for idx, image in enumerate(img_lst):
            if idx % 100 == 0:
                tf.compat.v1.logging.info('On image %d of %d', idx, len(img_lst))
            tf_example = dict_to_tf_example(image, label_map_dict, view_map_dict)
            writer.write(tf_example.SerializeToString())

        writer.close()


if __name__ == '__main__':
    tf.app.run()
