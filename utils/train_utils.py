# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utility functions for training."""

import six

import tensorflow as tf
from deeplab.core import preprocess_utils

slim = tf.contrib.slim

LOGITS_SCOPE_NAME = 'logits'
MERGED_LOGITS_SCOPE = 'merged_logits'
IMAGE_POOLING_SCOPE = 'image_pooling'
ASPP_SCOPE = 'aspp'
CONCAT_PROJECTION_SCOPE = 'concat_projection'
DECODER_SCOPE = 'decoder'
META_ARCHITECTURE_SCOPE = 'meta_architecture'


def add_softmax_cross_entropy_loss_for_each_scale(scales_to_logits,
                                                  labels,
                                                  num_classes,
                                                  ignore_label,
                                                  loss_weight=1.0,
                                                  upsample_logits=True,
                                                  scope=None):
    """Adds softmax cross entropy loss for logits of each scale.

    Args:
      scales_to_logits: A map from logits names for different scales to logits.
        The logits have shape [batch, logits_height, logits_width, num_classes].
      labels: Groundtruth labels with shape [batch, image_height, image_width, 1].
      num_classes: Integer, number of target classes.
      ignore_label: Integer, label to ignore.
      loss_weight: Float, loss weight.
      upsample_logits: Boolean, upsample logits or not.
      scope: String, the scope for the loss.

    Raises:
      ValueError: Label or logits is None.
    """
    if labels is None:
        raise ValueError('No label for softmax cross entropy loss.')

    for scale, logits in six.iteritems(scales_to_logits):
        loss_scope = None
        if scope:
            loss_scope = '%s_%s' % (scope, scale)

        if upsample_logits:
            # Label is not downsampled, and instead we upsample logits.
            logits = tf.image.resize_bilinear(
                logits,
                preprocess_utils.resolve_shape(labels, 4)[1:3],
                align_corners=True)
            scaled_labels = labels
        else:
            # Label is downsampled to the same size as logits.
            scaled_labels = tf.image.resize_nearest_neighbor(
                labels,
                preprocess_utils.resolve_shape(logits, 4)[1:3],
                align_corners=True)

        scaled_labels = tf.reshape(scaled_labels, shape=[-1])
        not_ignore_mask = tf.to_float(tf.not_equal(scaled_labels,
                                                   ignore_label)) * loss_weight
        one_hot_labels = slim.one_hot_encoding(
            scaled_labels, num_classes, on_value=1.0, off_value=0.0)
        tf.losses.softmax_cross_entropy(
            one_hot_labels,
            tf.reshape(logits, shape=[-1, num_classes]),
            weights=not_ignore_mask,
            scope=loss_scope)


def get_model_init_fn(train_logdir,
                      tf_initial_checkpoint,
                      initialize_last_layer,
                      last_layers,
                      ignore_missing_vars=False):
    """Gets the function initializing model variables from a checkpoint.

    Args:
      train_logdir: Log directory for training.
      tf_initial_checkpoint: TensorFlow checkpoint for initialization.
      initialize_last_layer: Initialize last layer or not.
      last_layers: Last layers of the model.
      ignore_missing_vars: Ignore missing variables in the checkpoint.

    Returns:
      Initialization function.
    """
    if tf_initial_checkpoint is None:
        tf.logging.info('Not initializing the model from a checkpoint.')
        return None

    if tf.train.latest_checkpoint(train_logdir):
        tf.logging.info('Ignoring initialization; other checkpoint exists')
        return None

    tf.logging.info('Initializing model from path: %s', tf_initial_checkpoint)

    # Variables that will not be restored.
    exclude_list = ['global_step']
    if not initialize_last_layer:
        exclude_list.extend(last_layers)

    variables_to_restore = slim.get_variables_to_restore(exclude=exclude_list)

    if variables_to_restore:
        return slim.assign_from_checkpoint_fn(
            tf_initial_checkpoint,
            variables_to_restore,
            ignore_missing_vars=ignore_missing_vars)
    return None


def get_model_gradient_multipliers(last_layers, last_layer_gradient_multiplier):
    """Gets the gradient multipliers.

    The gradient multipliers will adjust the learning rates for model
    variables. For the task of semantic segmentation, the models are
    usually fine-tuned from the models trained on the task of image
    classification. To fine-tune the models, we usually set larger (e.g.,
    10 times larger) learning rate for the parameters of last layer.

    Args:
      last_layers: Scopes of last layers.
      last_layer_gradient_multiplier: The gradient multiplier for last layers.

    Returns:
      The gradient multiplier map with variables as key, and multipliers as value.
    """
    gradient_multipliers = {}

    for var in slim.get_model_variables():
        # Double the learning rate for biases.
        if 'biases' in var.op.name:
            gradient_multipliers[var.op.name] = 2.

        # Use larger learning rate for last layer variables.
        for layer in last_layers:
            if layer in var.op.name and 'biases' in var.op.name:
                gradient_multipliers[var.op.name] = 2 * last_layer_gradient_multiplier
                break
            elif layer in var.op.name:
                gradient_multipliers[var.op.name] = last_layer_gradient_multiplier
                break

    return gradient_multipliers


def get_model_learning_rate(
        learning_policy, base_learning_rate, learning_rate_decay_step,
        learning_rate_decay_factor, training_number_of_steps, learning_power,
        slow_start_step, slow_start_learning_rate):
    """Gets model's learning rate.

    Computes the model's learning rate for different learning policy.
    Right now, only "step" and "poly" are supported.
    (1) The learning policy for "step" is computed as follows:
      current_learning_rate = base_learning_rate *
        learning_rate_decay_factor ^ (global_step / learning_rate_decay_step)
    See tf.train.exponential_decay for details.
    (2) The learning policy for "poly" is computed as follows:
      current_learning_rate = base_learning_rate *
        (1 - global_step / training_number_of_steps) ^ learning_power

    Args:
      learning_policy: Learning rate policy for training.
      base_learning_rate: The base learning rate for model training.
      learning_rate_decay_step: Decay the base learning rate at a fixed step.
      learning_rate_decay_factor: The rate to decay the base learning rate.
      training_number_of_steps: Number of steps for training.
      learning_power: Power used for 'poly' learning policy.
      slow_start_step: Training model with small learning rate for the first
        few steps.
      slow_start_learning_rate: The learning rate employed during slow start.

    Returns:
      Learning rate for the specified learning policy.

    Raises:
      ValueError: If learning policy is not recognized.
    """
    global_step = tf.train.get_or_create_global_step()
    if learning_policy == 'step':
        learning_rate = tf.train.exponential_decay(
            base_learning_rate,
            global_step,
            learning_rate_decay_step,
            learning_rate_decay_factor,
            staircase=True)
    elif learning_policy == 'poly':
        learning_rate = tf.train.polynomial_decay(
            base_learning_rate,
            global_step,
            training_number_of_steps,
            end_learning_rate=0,
            power=learning_power)
    else:
        raise ValueError('Unknown learning policy.')

    # Employ small learning rate at the first few steps for warm start.
    return tf.where(global_step < slow_start_step, slow_start_learning_rate,
                    learning_rate)


def _gather_loss(regularization_losses):
    """Gather the loss.

    Args:
      regularization_losses: Possibly empty list of regularization_losses
        to add to the clone losses.

    Returns:
      A tensor for the total loss.  Can be None.
    """
    # The return value.
    sum_loss = None
    # Individual components of the loss that will need summaries.
    clone_loss = None
    regularization_loss = None

    # Compute and aggregate losses on the clone device.
    # with tf.device(clone.device):
    all_losses = []
    losses = tf.get_collection(tf.GraphKeys.LOSSES)
    if losses:
        # clone_loss = tf.add_n(clone_losses, name='clone_loss')
        loss = tf.identity(losses, name='loss')
        # if num_clones > 1:
        #     clone_loss = tf.div(clone_loss, 1.0 * num_clones,
        #                         name='scaled_clone_loss')
        all_losses.append(loss)

    if regularization_losses:
        regularization_loss = tf.add_n(regularization_losses,
                                       name='regularization_loss')
        all_losses.append(regularization_loss)
    if all_losses:
        sum_loss = tf.add_n(all_losses)

    # Add the summaries out of the clone device block.
    if loss is not None:
        tf.summary.scalar('/'.join(filter(None,
                                          ['Losses', 'loss'])),
                          clone_loss)
    if regularization_loss is not None:
        tf.summary.scalar('Losses/regularization_loss', regularization_loss)
    return sum_loss


def _optimize(optimizer, clone, num_clones, regularization_losses,
              **kwargs):
    """Compute losses and gradients for a single clone.

    Args:
      optimizer: A tf.Optimizer  object.
      clone: A Clone namedtuple.
      num_clones: The number of clones being deployed.
      regularization_losses: Possibly empty list of regularization_losses
        to add to the clone losses.
      **kwargs: Dict of kwarg to pass to compute_gradients().

    Returns:
      A tuple (loss, grads_and_vars).
        - loss: A tensor for the total loss.  Can be None.
        - grads_and_vars: List of (gradient, variable). Can be empty.
    """
    sum_loss = _gather_loss(clone, num_clones, regularization_losses)
    grad = None
    if sum_loss is not None:
        grad = optimizer.compute_gradients(sum_loss, **kwargs)
    return sum_loss, grad


def _gradients(grad):
    """Calculate the sum gradient for each shared variable across all clones.

    This function assumes that the grad has been scaled appropriately by
    1 / num_clones.

    Args:
      grad: A List of tuples (gradient, variable)

    Returns:
       tuples of (gradient, variable)
    """
    # sum_grads = []
    # for grad_and_vars in zip(*clone_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad_var0_clone0, var0), ... (grad_varN_cloneN, varN))
    grads = []
    var = grad[0][1]
    for g, v in grad:
        assert v == var
        if g is not None:
            grads.append(g)
    if grads:
        if len(grads) > 1:
            sum_grad = tf.add_n(grads, name=var.op.name + '/sum_grads')
        else:
            sum_grad = grads[0]
        # sum_grads.append((sum_grad, var))
    return (sum_grad, var)


def optimize(optimizer, regularization_losses=None, **kwargs):
    """Compute losses and gradients

    # Note: The regularization_losses are added to the first clone losses.

    Args:
     # clones: List of `Clones` created by `create_clones()`.
     optimizer: An `Optimizer` object.
     regularization_losses: Optional list of regularization losses. If None it
       will gather them from tf.GraphKeys.REGULARIZATION_LOSSES. Pass `[]` to
       exclude them.
     **kwargs: Optional list of keyword arguments to pass to `compute_gradients`.

    Returns:
     A tuple (total_loss, grads_and_vars).
       - total_loss: A Tensor containing the average of the clone losses including
         the regularization loss.
       - grads_and_vars: A List of tuples (gradient, variable) containing the sum
         of the gradients for each variable.

    """
    # grads_and_vars = []
    # clones_losses = []
    # num_clones = len(clones)
    if regularization_losses is None:
        regularization_losses = tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES)
    # for clone in clones:
    #     with tf.name_scope(clone.scope):
    #         clone_loss, clone_grad = _optimize_clone(
    #             optimizer, clone, num_clones, regularization_losses, **kwargs)
    #         if clone_loss is not None:
    #             clones_losses.append(clone_loss)
    #             grads_and_vars.append(clone_grad)
    #         # Only use regularization_losses for the first clone
    #         regularization_losses = None

    loss, grad = _optimize(optimizer,
                           regularization_losses,
                           **kwargs)

    # if clone_loss is not None:
    #     clones_losses.append(clone_loss)
    #     grads_and_vars.append(clone_grad)
    # Only use regularization_losses for the first clone
    regularization_losses = None

    # Compute the total_loss summing all the clones_losses.
    # total_loss = tf.add_n(clones_losses, name='total_loss')
    total_loss = tf.identity(loss, name='total_loss')
    # Sum the gradients across clones.
    grads_and_vars = _gradients(grad)

    return total_loss, grads_and_vars


def get_extra_layer_scopes(last_layers_contain_logits_only=False):
    """Gets the scopes for extra layers.

    Args:
      last_layers_contain_logits_only: Boolean, True if only consider logits as
      the last layer (i.e., exclude ASPP module, decoder module and so on)

    Returns:
      A list of scopes for extra layers.
    """
    if last_layers_contain_logits_only:
        return [LOGITS_SCOPE_NAME]
    else:
        return [
            LOGITS_SCOPE_NAME,
            IMAGE_POOLING_SCOPE,
            ASPP_SCOPE,
            CONCAT_PROJECTION_SCOPE,
            DECODER_SCOPE,
            META_ARCHITECTURE_SCOPE,
        ]
