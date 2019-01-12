import numpy as np
import cv2
import os
import random

import tensorflow as tf


class Data(object):

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.label_to_index = {}
        self._prepare_data()


    def get_data(self):
        return self.data_index


    def get_data_size(self):
        num = 0
        for key, val in self.data_index.items():
            num += len(val)

        return num


    def get_label_to_index(self):
        return self.label_to_index


    def _prepare_data(self):
        classes = os.listdir(self.dataset_dir)
        classes.sort()
        tf.logging.info("classes: %s", classes)

        for index, cls in enumerate(classes):
            self.label_to_index[cls] = index

        self.data_index = {}
        for cls in classes:
            l_train_path = os.path.join(self.dataset_dir, cls, 'train')
            imgs = os.listdir(l_train_path)
            # imgs.sort()
            self.data_index[cls] = []
            for img in imgs:
                views_path = os.path.join(l_train_path, img)
                views = os.listdir(views_path)
                v_paths = ''
                for v in views:
                    v_path = os.path.join(views_path, v)
                    v_paths += v_path + '|'

                self.data_index[cls].append({'view_paths':v_paths[:-1], 'label':cls})

        tf.logging.info("data prepared.")


class DataLoader(object):
    """
    Wrapper class around the new Tensorflows dataset pipeline.

    Handles loading, partitioning, and preparing training data.
    """

    def __init__(self, dataset, batch_size, height, weight, shuffle=True):
        self.resize_h = height
        self.resize_w = weight

        images, labels = self._get_data(dataset.get_data(),
                                        batch_size,
                                        dataset.get_label_to_index())

        # self.data_size = dataset.get_data_size() // batch_size
        self.data_size = len(images)

        # create dataset, Creating a source
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.map(self._parse_func, num_parallel_calls=8)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=(int(self.data_size * 0.4) + 3 * 1))
        dataset = dataset.batch(1)
        self.dataset = dataset.repeat()


    def _get_data(self, data, batch_size, label_to_index):
        images = []
        labels = []

        max = int(batch_size / 2)
        for i in range(max):
            for _, cls_lst in data.items():
                random.shuffle(cls_lst)
                n = len(cls_lst)
                tf.logging.info("num : " + str(n) + ", cls : " + str(cls_lst[0]['label']))
                batch_img = []
                batch_lbl = []
                for idx, e in enumerate(cls_lst):
                    batch_img.append(e['view_paths'].split('|'))
                    batch_lbl.append(label_to_index[e['label']])

                    if idx % batch_size == (batch_size-1):
                        images.append(batch_img)
                        labels.append(batch_lbl)
                        batch_img = []
                        batch_lbl = []
                    elif idx == n-1:
                        batch_num = len(batch_img)
                        for i in range(batch_size-batch_num):
                            batch_img.append(cls_lst[i]['view_paths'].split('|'))
                            batch_lbl.append(label_to_index[cls_lst[i]['label']])
                        images.append(batch_img)
                        labels.append(batch_lbl)
                        batch_img = []
                        batch_lbl = []

        return images, labels


    def _parse_func(self, image_paths, labels):
        image_paths_lst = tf.unstack(image_paths)
        i = []
        for image_path in image_paths_lst:
            view_paths_lst = tf.unstack(image_path)
            v = []
            for v_path in view_paths_lst:
                image_string = tf.read_file(v_path)
                image_decoded = tf.image.decode_png(image_string, channels=3)
                # image_decoded = tf.image.decode_jpeg(image_string, channels=3)
                # cropped_image = tf.image.central_crop(image_decoded, 0.7)
                # rotated_image = tf.image.rot90(image_decoded, 1)
                resized_image = tf.image.resize_images(image_decoded,
                                                       [self.resize_h, self.resize_w])
                # image = tf.cast(image_decoded, tf.float32)
                image = tf.image.convert_image_dtype(resized_image, dtype=tf.float32)
                # Finally, rescale to [-1,1] instead of [0, 1)
                # image = tf.subtract(image, 0.5)
                # image = tf.multiply(image, 2.0)
                v.append(image)

            i.append(v)

        return i, labels


    def get_data_size(self):
        return self.data_size
