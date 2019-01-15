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


def _gather_loss(regularization_losses, scope):
    """Gather the loss.

    Args:
      regularization_losses: Possibly empty list of regularization_losses
        to add to the losses.

    Returns:
      A tensor for the total loss.  Can be None.
    """

    # The return value.
    sum_loss = None
    # Individual components of the loss that will need summaries.
    loss = None
    regularization_loss = None

    # Compute and aggregate losses on the clone device.
    all_losses = []
    losses = tf.get_collection(tf.GraphKeys.LOSSES, scope)
    if losses:
        loss = tf.add_n(losses, name='losses')
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
                          loss)
    if regularization_loss is not None:
        tf.summary.scalar('Losses/regularization_loss', regularization_loss)
    return sum_loss


def _optimize(optimizer, regularization_losses, scope, **kwargs):
    """Compute losses and gradients.

    Args:
      optimizer: A tf.Optimizer  object.
      regularization_losses: Possibly empty list of regularization_losses
        to add to the losses.
      **kwargs: Dict of kwarg to pass to compute_gradients().

    Returns:
      A tuple (loss, grads_and_vars).
        - loss: A tensor for the total loss.  Can be None.
        - grads_and_vars: List of (gradient, variable). Can be empty.
    """
    sum_loss = _gather_loss(regularization_losses, scope)
    grad = None
    if sum_loss is not None:
        grad = optimizer.compute_gradients(sum_loss, **kwargs)
    return sum_loss, grad


def _gradients(grad):
    """Calculate the sum gradient for each shared variable across all clones.

    This function assumes that the grad has been scaled appropriately by
    1 / num_clones.

    Args:
      grad: A List of List of tuples (gradient, variable)

    Returns:
       tuples of (gradient, variable)
    """
    sum_grads = []
    for grad_and_vars in zip(*grad):
        # Note that each grad_and_vars looks like the following:
        #   ((grad_var0_clone0, var0), ... (grad_varN_cloneN, varN))
        grads = []
        var = grad_and_vars[0][1]
        for g, v in grad_and_vars:
            assert v == var
            if g is not None:
                grads.append(g)
        if grads:
            if len(grads) > 1:
                sum_grad = tf.add_n(grads, name=var.op.name + '/sum_grads')
            else:
                sum_grad = grads[0]
            sum_grads.append((sum_grad, var))

    return sum_grads


def optimize(optimizer, scope=None, regularization_losses=None, **kwargs):
    """Compute losses and gradients

    # Note: The regularization_losses are added to losses.

    Args:
     optimizer: An `Optimizer` object.
     regularization_losses: Optional list of regularization losses. If None it
       will gather them from tf.GraphKeys.REGULARIZATION_LOSSES. Pass `[]` to
       exclude them.
     **kwargs: Optional list of keyword arguments to pass to `compute_gradients`.

    Returns:
     A tuple (total_loss, grads_and_vars).
       - total_loss: A Tensor containing the average of the losses including
         the regularization loss.
       - grads_and_vars: A List of tuples (gradient, variable) containing the sum
         of the gradients for each variable.

    """

    grads_and_vars = []
    losses = []
    if regularization_losses is None:
        regularization_losses = tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES, scope)
    # with tf.name_scope(scope):
    loss, grad = _optimize(optimizer,
                           regularization_losses,
                           scope,
                           **kwargs)
    if loss is not None:
        losses.append(loss)
        grads_and_vars.append(grad)
    # Only use regularization_losses for the first clone
    regularization_losses = None

    # Compute the total_loss summing all the losses.
    total_loss = tf.add_n(losses, name='total_loss')
    # Sum the gradients across clones.
    grads_and_vars = _gradients(grads_and_vars)

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


def edit_trainable_variables(removed):
    # gets a reference to the list containing the trainable variables
    trainable_collection = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
    variables_to_remove = []
    for var in trainable_collection:
        # uses the attribute 'name' of the variable
        if var.name.startswith(removed):
            variables_to_remove.append(var)
    for rem in variables_to_remove:
        trainable_collection.remove(rem)


def add_variables_summaries(learning_rate):
    summaries = []
    for variable in slim.get_model_variables():
        summaries.append(tf.summary.histogram(variable.op.name, variable))
    summaries.append(tf.summary.scalar('training/Learning Rate', learning_rate))
    return summaries


def restore_fn(flags):
    """Returns a function run by the chief worker to warm-start the training.
    Note that the init_fn is only run when initializing the model during the very
    first global step.

    """
    # if flags.tf_initial_checkpoint is None:
    #     return None

    # Warn the user if a checkpoint exists in the train_dir. Then ignore.
    # if tf.train.latest_checkpoint(flags.train_dir):
    #     tf.logging.info(
    #         'Ignoring --checkpoint_path because a checkpoint already exists in %s'
    #         % flags.train_dir)
    #     return None

    exclusions = []
    if flags.checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in flags.checkpoint_exclude_scopes.split(',')]

    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    # Change model scope if necessary.
    if flags.checkpoint_model_scope is not None:
        variables_to_restore = \
            {var.op.name.replace(flags.model_name,
                                 flags.checkpoint_model_scope): var
             for var in variables_to_restore}


    tf.logging.info('Fine-tuning from %s. Ignoring missing vars: %s' %
                    (flags.pre_trained_checkpoint, flags.ignore_missing_vars))
    slim.assign_from_checkpoint_fn(flags.pre_trained_checkpoint,
                                   variables_to_restore,
                                   ignore_missing_vars=flags.ignore_missing_vars)


def get_variables_to_train(flags):
    """Returns a list of variables to train.

    Returns:
      A list of variables to train by the optimizer.
    """
    if flags.trainable_scopes is None:
        # print(tf.trainable_variables())
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in flags.trainable_scopes.split(',')]

    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train
