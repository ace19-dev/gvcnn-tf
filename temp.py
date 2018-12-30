import tensorflow as tf

def _gather_clone_loss(clone, num_clones, regularization_losses):
    """Gather the loss for a single clone.

    Args:
      clone: A Clone namedtuple.
      num_clones: The number of clones being deployed.
      regularization_losses: Possibly empty list of regularization_losses
        to add to the clone losses.

    Returns:
      A tensor for the total loss for the clone.  Can be None.
    """
    # The return value.
    sum_loss = None
    # Individual components of the loss that will need summaries.
    clone_loss = None
    regularization_loss = None
    # Compute and aggregate losses on the clone device.
    with tf.device(clone.device):
        all_losses = []
        clone_losses = tf.get_collection(tf.GraphKeys.LOSSES, clone.scope)
        if clone_losses:
            clone_loss = tf.add_n(clone_losses, name='clone_loss')
            if num_clones > 1:
                clone_loss = tf.div(clone_loss, 1.0 * num_clones,
                                    name='scaled_clone_loss')
            all_losses.append(clone_loss)
        if regularization_losses:
            regularization_loss = tf.add_n(regularization_losses,
                                           name='regularization_loss')
            all_losses.append(regularization_loss)
        if all_losses:
            sum_loss = tf.add_n(all_losses)
    # Add the summaries out of the clone device block.
    if clone_loss is not None:
        tf.summary.scalar('/'.join(filter(None,
                                          ['Losses', clone.scope, 'clone_loss'])),
                          clone_loss)
    if regularization_loss is not None:
        tf.summary.scalar('Losses/regularization_loss', regularization_loss)
    return sum_loss


def _optimize_clone(optimizer, clone, num_clones, regularization_losses,
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
      A tuple (clone_loss, clone_grads_and_vars).
        - clone_loss: A tensor for the total loss for the clone.  Can be None.
        - clone_grads_and_vars: List of (gradient, variable) for the clone.
          Can be empty.
    """
    sum_loss = _gather_clone_loss(clone, num_clones, regularization_losses)
    clone_grad = None
    if sum_loss is not None:
        with tf.device(clone.device):
            clone_grad = optimizer.compute_gradients(sum_loss, **kwargs)
    return sum_loss, clone_grad


def optimize_clones(clones, optimizer,
                    regularization_losses=None,
                    **kwargs):
    """Compute clone losses and gradients for the given list of `Clones`.

    Note: The regularization_losses are added to the first clone losses.

    Args:
     clones: List of `Clones` created by `create_clones()`.
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
    grads_and_vars = []
    clones_losses = []
    num_clones = len(clones)
    if regularization_losses is None:
        regularization_losses = tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES)
    for clone in clones:
        with tf.name_scope(clone.scope):
            clone_loss, clone_grad = _optimize_clone(
                optimizer, clone, num_clones, regularization_losses, **kwargs)
            if clone_loss is not None:
                clones_losses.append(clone_loss)
                grads_and_vars.append(clone_grad)
            # Only use regularization_losses for the first clone
            regularization_losses = None
    # Compute the total_loss summing all the clones_losses.
    total_loss = tf.add_n(clones_losses, name='total_loss')
    # Sum the gradients across clones.
    grads_and_vars = _sum_clones_gradients(grads_and_vars)
    return total_loss, grads_and_vars


def _sum_clones_gradients(clone_grads):
    """Calculate the sum gradient for each shared variable across all clones.

    This function assumes that the clone_grads has been scaled appropriately by
    1 / num_clones.

    Args:
      clone_grads: A List of List of tuples (gradient, variable), one list per
      `Clone`.

    Returns:
       List of tuples of (gradient, variable) where the gradient has been summed
       across all clones.
    """
    sum_grads = []
    for grad_and_vars in zip(*clone_grads):
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


def _add_gradients_summaries(grads_and_vars):
    """Add histogram summaries to gradients.

    Note: The summaries are also added to the SUMMARIES collection.

    Args:
      grads_and_vars: A list of gradient to variable pairs (tuples).

    Returns:
      The _list_ of the added summaries for grads_and_vars.
    """
    summaries = []
    for grad, var in grads_and_vars:
        if grad is not None:
            if isinstance(grad, tf.IndexedSlices):
                grad_values = grad.values
            else:
                grad_values = grad
            summaries.append(tf.summary.histogram(var.op.name + ':gradient',
                                                  grad_values))
            summaries.append(tf.summary.histogram(var.op.name + ':gradient_norm',
                                                  tf.global_norm([grad_values])))
        else:
            tf.logging.info('Var %s has no gradient', var.op.name)
    return summaries
