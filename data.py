import tensorflow as tf

import random


NUM_VIEWS = 8

class Dataset(object):
    """
    Wrapper class around the new Tensorflows dataset pipeline.

    Handles loading, partitioning, and preparing training data.
    """

    def __init__(self, tfrecord_path, height, width, batch_size=1):
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

        # Prefetches a batch at a time to smooth out the time taken to load input
        # files for shuffling and processing.
        dataset = dataset.prefetch(buffer_size=batch_size)
        # The shuffle transformation uses a finite-sized buffer to shuffle elements
        # in memory. The parameter is the number of elements in the buffer. For
        # completely uniform shuffling, set the parameter to be the same as the
        # number of elements in the dataset.
        dataset = dataset.shuffle(1000 + 3 * batch_size)
        dataset = dataset.repeat()
        self.dataset = dataset.batch(batch_size)


    def decode(self, serialized_example):
        """Parses an image and label from the given `serialized_example`."""
        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                # 'image/filename': tf.FixedLenFeature([], tf.string),
                'image/encoded': tf.FixedLenFeature([NUM_VIEWS], tf.string),
                'image/label': tf.FixedLenFeature([], tf.int64),
            })

        images = []
        # labels = []
        img_lst = tf.unstack(features['image/encoded'])
        # lbl_lst = tf.unstack(features['image/label'])
        for i, img in enumerate(img_lst):
            # Convert from a scalar string tensor to a float32 tensor with shape
            image_decoded = tf.image.decode_png(img, channels=3)
            image = tf.image.resize_images(image_decoded, [self.resize_h, self.resize_w])
            images.append(image)
            # labels.append(lbl_lst[i])

        # Convert label from a scalar uint8 tensor to an int32 scalar.
        label = features['image/label']

        return images, label

    def augment(self, images, label):
        """Placeholder for data augmentation."""
        # OPTIONAL: Could reshape into a 28x28 image and apply distortions
        # here.  Since we are not applying any distortions in this
        # example, and the next step expects the image to be flattened
        # into a vector, we don't bother.
        img_lst = []
        img_tensor_lst = tf.unstack(images)
        for i, image in enumerate(img_tensor_lst):
            image = tf.image.central_crop(image, 0.9)
            image = tf.image.random_flip_up_down(image)
            image = tf.image.random_flip_left_right(image)
            image = tf.image.rot90(image, k=random.randint(0, 4))
            paddings = tf.constant([[14, 14], [14, 14], [0, 0]])  # 299
            image = tf.pad(image, paddings, "CONSTANT")

            img_lst.append(image)

        return img_lst, label


    def normalize(self, images, label):
        """Convert `image` from [0, 255] -> [-0.5, 0.5] floats."""
        img_lst = []
        img_tensor_lst = tf.unstack(images)
        for i, image in enumerate(img_tensor_lst):
            image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
            img_lst.append(image)

        return img_lst, label
