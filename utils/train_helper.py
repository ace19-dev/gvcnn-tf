import sys
import logging
import tensorflow as tf


def allreduce_grads(all_grads, average=True):
    """
    REFERENCE : https://github.com/ppwwyyxx/tensorpack/blob/83e4e187af5765792408e7b7163efd4744d63628/tensorpack/graph_builder/utils.py
    All-reduce average the gradients among K devices. Results are broadcasted to all devices.
    Args:
        all_grads (K x N): List of list of gradients. N is the number of variables.
        average (bool): average gradients or not.
    Returns:
        K x N: same as input, but each grad is replaced by the average over K devices.
    """
    # from tensorflow.contrib import nccl
    from tensorflow.python.ops import nccl_ops
    nr_tower = len(all_grads)
    if nr_tower == 1:
        return all_grads
    new_all_grads = []  # N x K
    for grads in zip(*all_grads):
        summed = nccl_ops.all_sum(grads)

        grads_for_devices = []  # K
        for g in summed:
            with tf.device(g.device):
                # tensorflow/benchmarks didn't average gradients
                if average:
                    g = tf.multiply(g, 1.0 / nr_tower, name='allreduce_avg')
            grads_for_devices.append(g)
        new_all_grads.append(grads_for_devices)

    # transpose to K x N
    ret = list(zip(*new_all_grads))
    return ret


def split_grad_list(grad_list):
    """
    Args:
        grad_list: K x N x 2
    Returns:
        K x N: gradients
        K x N: variables
    """
    g = []
    v = []
    for tower in grad_list:
        g.append([x[0] for x in tower])
        v.append([x[1] for x in tower])
    return g, v


def merge_grad_list(all_grads, all_vars):
    """
    Args:
        all_grads (K x N): gradients
        all_vars(K x N): variables
    Return:
        K x N x 2: list of list of (grad, var) pairs
    """
    return [list(zip(gs, vs)) for gs, vs in zip(all_grads, all_vars)]


def get_post_init_ops():
    """
    Copy values of variables on GPU 0 to other GPUs.
    """
    # literally all variables, because it's better to sync optimizer-internal variables as well
    all_vars = tf.global_variables() + tf.local_variables()
    var_by_name = dict([(v.name, v) for v in all_vars])
    post_init_ops = []
    for v in all_vars:
        if not v.name.startswith('tower'):
            continue
        if v.name.startswith('tower0'):
            # no need for copy to tower0
            continue
        # in this trainer, the master name doesn't have the towerx/ prefix
        split_name = v.name.split('/')
        prefix = split_name[0]
        realname = '/'.join(split_name[1:])
        if prefix in realname:
            # logger.warning("variable {} has its prefix {} appears multiple times in its name!".format(v.name, prefix))
            pass
        copy_from = var_by_name.get(v.name.replace(prefix, 'tower0'))
        if copy_from is not None:
            post_init_ops.append(v.assign(copy_from.read_value()))
        else:
            print("Cannot find {} in the graph!".format(realname))

    print("'sync_variables_from_main_tower' includes {} operations.".format(len(post_init_ops)))
    return tf.group(*post_init_ops, name='sync_variables_from_main_tower')
