import numpy as np
import cv2
import os
import random

import tensorflow as tf


# class Data(object):
#
#     def __init__(self, dataset_dir):
#         self.dataset_dir = dataset_dir
#         self.label_to_index = {}
#         self._prepare_data()
#
#
#     def get_data(self):
#         return self.data_index
#
#
#     def get_data_size(self):
#         num = 0
#         for key, val in self.data_index.items():
#             num += len(val)
#
#         return num
#
#
#     def get_label_to_index(self):
#         return self.label_to_index
#
#
#     def _prepare_data(self):
#         classes = os.listdir(self.dataset_dir)
#         classes.sort()
#         tf.logging.info("classes: %s", classes)
#
#         for index, cls in enumerate(classes):
#             self.label_to_index[cls] = index
#
#         self.data_index = {}
#         for cls in classes:
#             l_train_path = os.path.join(self.dataset_dir, cls, 'train')
#             imgs = os.listdir(l_train_path)
#             # imgs.sort()
#             self.data_index[cls] = []
#             for img in imgs:
#                 views_path = os.path.join(l_train_path, img)
#                 views = os.listdir(views_path)
#                 v_paths = ''
#                 for v in views:
#                     v_path = os.path.join(views_path, v)
#                     v_paths += v_path + '|'
#
#                 self.data_index[cls].append({'view_paths':v_paths[:-1], 'label':cls})
#
#         tf.logging.info("data prepared.")
#

class Dataset(object):
    """
    Wrapper class around the new Tensorflows dataset pipeline.

    Handles loading, partitioning, and preparing training data.
    """

    def __init__(self, tfrecord_path, batch_size, num_epochs, height, width):
        self.resize_h = height
        self.resize_w = width

        dataset = tf.data.TFRecordDataset(tfrecord_path,
                                          compression_type='GZIP',
                                          num_parallel_reads=batch_size * 4)

        # dataset = dataset.map(self._parse_func, num_parallel_calls=8)
        # The map transformation takes a function and applies it to every element
        # of the dataset.
        dataset = dataset.map(self.decode, num_parallel_calls=8)
        dataset = dataset.map(self.augment, num_parallel_calls=8)
        dataset = dataset.map(self.normalize, num_parallel_calls=8)

        # The shuffle transformation uses a finite-sized buffer to shuffle elements
        # in memory. The parameter is the number of elements in the buffer. For
        # completely uniform shuffling, set the parameter to be the same as the
        # number of elements in the dataset.
        dataset = dataset.shuffle(1000 + 3 * batch_size)
        dataset = dataset.repeat(num_epochs)
        self.dataset = dataset.batch(batch_size)

    def decode(self, serialized_example):
        """Parses an image and label from the given `serialized_example`."""
        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                # 'image/filename': tf.FixedLenFeature([], tf.string),
                'image/encoded': tf.FixedLenFeature([], tf.string),
                'image/label': tf.FixedLenFeature([], tf.int64),
            })

        # Convert from a scalar string tensor to a float32 tensor with shape
        image_decoded = tf.image.decode_png(features['image/encoded'], channels=3)
        image = tf.image.resize_images(image_decoded, [self.resize_h, self.resize_w])

        # Convert label from a scalar uint8 tensor to an int32 scalar.
        label = tf.cast(features['image/label'], tf.int64)

        return image, label

    def augment(self, image, label):
        """Placeholder for data augmentation."""
        # OPTIONAL: Could reshape into a 28x28 image and apply distortions
        # here.  Since we are not applying any distortions in this
        # example, and the next step expects the image to be flattened
        # into a vector, we don't bother.
        return image, label

    def normalize(self, image, label):
        """Convert `image` from [0, 255] -> [-0.5, 0.5] floats."""
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
        return image, label
