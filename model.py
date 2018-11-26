from slim.nets import inception_v4


def FCN(inputs,
        num_classes=1000,
        is_training=True,
        dropout_keep_prob=0.8,
        scope='GVCNN'):

    """
    Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer
      is omitted and the input features to the logits layer (before dropout)
      are returned instead.
    is_training: whether is training or not.
    dropout_keep_prob: the percentage of activation values that are retained.


    Returns:
        net: a Tensor with the logits (pre-softmax activations) if num_classes
        is a non - zero integer, or the non - dropped - out input to the logits
        layer if num_classes is 0 or None.
        end_points: a dictionary from components of the network to the corresponding activation.
    """





def FC():





def raw_view_descriptors():





def view_pooling():





def group_fusion():



def gvcnn(inputs,
          num_classes=1000,
          is_training=True,
          dropout_keep_prob=0.8,
          scope='GVCNN'):
    """
    Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer
      is omitted and the input features to the logits layer (before dropout)
      are returned instead.
    is_training: whether is training or not.
    dropout_keep_prob: the percentage of activation values that are retained.


    Returns:
        net: a Tensor with the logits (pre-softmax activations) if num_classes
        is a non - zero integer, or the non - dropped - out input to the logits
        layer if num_classes is 0 or None.
        end_points: a dictionary from components of the network to the corresponding activation.
    """
    return logits, end_points
