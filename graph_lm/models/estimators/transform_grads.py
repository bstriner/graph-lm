import tensorflow as tf
from tensorflow.contrib.slim.python.slim.learning import clip_gradient_norms


def clip_gradient_values(gradients_to_variables, min_value, max_value):
    clipped_grads_and_vars = []
    for grad, var in gradients_to_variables:
        if grad is not None:
            if isinstance(grad, tf.IndexedSlices):
                tmp = tf.clip_by_value(grad.values, min_value, max_value)
                grad = tf.IndexedSlices(tmp, grad.indices, grad.dense_shape)
            else:
                grad = tf.clip_by_value(grad, min_value, max_value)
        clipped_grads_and_vars.append((grad, var))
    return clipped_grads_and_vars


def make_transform_grads_fn(params):
    def transform_grads_fn(grads):
        # Clip gradients.
        if params.clip_gradient_norm > 0 and params.clip_gradient_value > 0:
            raise ValueError("Only one of clip_gradient_norm or clip_gradient_value should be set")
        if params.clip_gradient_norm > 0:
            with tf.name_scope('clip_grads'):
                grads = clip_gradient_norms(grads, params.clip_gradient_norm)
        if params.clip_gradient_value > 0:
            with tf.name_scope('clip_grads'):
                grads = clip_gradient_values(grads, -params.clip_gradient_value, params.clip_gradient_value)
        return grads

    return transform_grads_fn
