
import tensorflow as tf


def main(unused_argv):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    with tf.Graph().as_default() as graph:
        # x = tf.constant([[8, 1, 220, 55], [3, 4, 3, -1]])
        # x_max = tf.reduce_max(x, axis=0)
        #
        # y = tf.constant([[9, 9, 300, 8], [5, 5, -3, 0]])
        # x_y = tf.stack([x,y])
        #
        # x_y_max = tf.reduce_max(x_y, axis=0)
        # _max = tf.math.maximum(x, y)

        final_view_descriptors = tf.constant([[8, 1, 220, 55], [3, 4, 3, -1], [54, 1, 6, -53], [-3, -4, 35, -1], [0, 34, 0, -23]])
        group_scheme = tf.constant([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 1, 1], [0, 0, 0, 0, 0]])

        group_descriptors = {}
        dummy = tf.zeros_like(final_view_descriptors[0])

        scheme_list = tf.unstack(group_scheme)
        indices = [tf.squeeze(tf.where(elem), axis=1) for elem in scheme_list]
        for i, ind in enumerate(indices):
            pooled_view = tf.cond(tf.greater(tf.size(ind), 0),
                                  lambda: tf.gather(final_view_descriptors, ind),
                                  lambda: tf.expand_dims(dummy, 0))

            group_descriptors[i] = tf.reduce_mean(pooled_view, axis=0)

        sess_config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
        with tf.compat.v1.Session(config=sess_config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())

            # print(sess.run(x_max))
            # print(sess.run(x_y_max))
            # print(sess.run(_max))

            print(sess.run(group_descriptors))



if __name__ == '__main__':
    tf.compat.v1.app.run()