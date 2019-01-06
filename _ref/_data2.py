import numpy as np
import cv2
import os
import random

import tensorflow as tf


class Data(object):

    def __init__(self, dataset_dir, h_w):
        self.dataset_dir = dataset_dir
        self.resize_h = h_w[0]
        self.resize_w = h_w[1]

        self.cls = []
        self.label_to_index = {}
        self.cls_start_step = {}
        self.images, self.labels = self._prepare_data()
        self.shuffle_all()


    def _prepare_data(self):
        classes = os.listdir(self.dataset_dir)
        classes.sort()
        tf.logging.info("classes: %s", classes)

        labels_index = {}
        for index, cls in enumerate(classes):
            labels_index[cls] = index
            self.cls.append(index)
            self.label_to_index[index] = cls
            self.cls_start_step[index] = 0

        images = {}
        labels = {}
        for cls in classes:
            cls_images = []
            cls_labels = []
            l_train_path = os.path.join(self.dataset_dir, cls, 'train')
            imgs = os.listdir(l_train_path)
            # self.cls_data_size[cls] = len(imgs)
            # imgs.sort()
            for img in imgs:
                views_path = os.path.join(l_train_path, img)
                views = os.listdir(views_path)
                vp_lst = []
                # ls = []
                for v in views:
                    v_path = os.path.join(views_path, v)
                    vp_lst.append(v_path)
                    # ls.append(cls)

                cls_images.append(vp_lst)
                cls_labels.append(labels_index[cls])

            images[labels_index[cls]] = cls_images
            labels[labels_index[cls]] = cls_labels

        return images, labels


    def shuffle_all(self):
        for key, value in self.images.items():
            combined = list(zip(self.images[key], self.labels[key]))
            random.shuffle(combined)
            self.images[key], self.labels[key] = zip(*combined)


    def _shuffle(self, selected):
        combined = list(zip(self.images[selected], self.labels[selected]))
        random.shuffle(combined)
        self.images[selected], self.labels[selected] = zip(*combined)


    def _parse_function(self, start, end, selected):
        images = []
        # labels = []
        batch = self.images[selected][start:end]
        for idx, views_path in enumerate(batch):
            views_path.sort()
            views = []
            for view in views_path:
                image_string = tf.read_file(view)
                # image_decoded = tf.image.decode_png(image_string, channels=3)
                image_decoded = tf.image.decode_png(image_string, channels=3)
                # cropped_image = tf.image.central_crop(image_decoded, 0.7)
                # rotated_image = tf.image.rot90(image_decoded, 1)
                resized_image = tf.image.resize_images(image_decoded, [self.resize_h, self.resize_w])
                # image = tf.cast(image_decoded, tf.float32)
                # image = tf.image.convert_image_dtype(resized_image, dtype=tf.float32)
                # Finally, rescale to [-1,1] instead of [0, 1)
                # image = tf.subtract(image, 0.5)
                # image = tf.multiply(image, 2.0)
                views.append(resized_image)

            images.append(views)

        # labels = tf.convert_to_tensor(self.labels[start:end], dtype=tf.string)
        labels = (self.labels[selected][start:end])

        return tf.stack(images, 0), labels


    def next_batch(self, batch_size):
        # selected = random.choice(self.cls)

        batches_x = []
        batches_y = []
        for selected in self.cls:
            start = self.cls_start_step[selected]
            end = start + batch_size
            if end > self.data_size(selected):
                self._shuffle(selected)
                start = 0
                end = start + batch_size
                self.cls_start_step[selected] = end

            batch_x, batch_y = self._parse_function(start, end, selected)
            batches_x.append(batch_x)
            batches_y.append(batch_y)

        x = tf.convert_to_tensor(batches_x)
        y = tf.reshape(batches_y, [-1])

        return x, y


    def size(self):
        sum = 0
        for idx, _ in enumerate(self.cls):
            num = len(self.images[idx])
            sum += num

        return sum / len(self.cls)


    def data_size(self, selected):
        return len(self.images[selected])


    def get_label_to_index(self):
        return self.label_to_index
