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

import PIL.Image
import tensorflow as tf

from dataset_tools import dataset_util


RANDOM_SEED = 8045

flags = tf.app.flags
flags.DEFINE_string('dataset_dir',
                    '/home/ace19/dl_data/modelnet',
                    'Root Directory to raw modelnet dataset.')
flags.DEFINE_string('output_path',
                    '/home/ace19/dl_data/modelnet/test.record',
                    'Path to output TFRecord')
flags.DEFINE_string('dataset_category',
                    'test',
                    'dataset category, train or test')

FLAGS = flags.FLAGS


def get_data_map_dict(label_to_index):
    label_map_dict = {}
    view_map_dict = {}

    cls_lst = os.listdir(FLAGS.dataset_dir)
    for i, label in enumerate(cls_lst):
        label_path = os.path.join(FLAGS.dataset_dir, label, FLAGS.dataset_category)
        if not os.path.isdir(label_path):
            continue
        img_lst = os.listdir(label_path)
        for n, img in enumerate(img_lst):
            img_path = os.path.join(label_path, img)
            view_lst = os.listdir(img_path)
            views = []
            for k, view in enumerate(view_lst):
                v_path = os.path.join(img_path, view)
                views.append(v_path)
            label_map_dict[img] = label_to_index[label]
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
    # labels = []
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
    tf.logging.set_verbosity(tf.logging.INFO)

    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path, options=options)

    dataset_lst = os.listdir(FLAGS.dataset_dir)
    dataset_lst.sort()
    label_to_index = {cls: i for i, cls in enumerate(dataset_lst)}
    label_map_dict, view_map_dict = get_data_map_dict(label_to_index)

    tf.logging.info('Reading from modelnet dataset.')
    cls_lst = os.listdir(FLAGS.dataset_dir)
    for i, label in enumerate(cls_lst):
        label_path = os.path.join(FLAGS.dataset_dir, label, FLAGS.dataset_category)
        if not os.path.isdir(label_path):
            continue
        img_lst = os.listdir(label_path)
        for idx, image in enumerate(img_lst):
            if idx % 100 == 0:
                tf.logging.info('On image %d of %d', idx, len(img_lst))
            tf_example = dict_to_tf_example(image, label_map_dict, view_map_dict)
            writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()
