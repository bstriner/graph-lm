import numpy as np
import tensorflow as tf
from tensorflow.python.framework import common_shapes


def power_iter(u, w):
    vt = tf.nn.l2_normalize(tf.matmul(tf.transpose(w, (1, 0)), u))
    ut = tf.nn.l2_normalize(tf.matmul(w, vt))
    return ut, vt


# def spec_norm(u, w, v):
#    s = tf.matmul(tf.matmul(tf.transpose(u, (1, 0)), w), v)
#    ws = w / s
#    return ws

def spec_norm(u, w):
    return tf.norm(tf.matmul(tf.transpose(w, (1, 0)), u), ord=2)


def sn_calc(w, scope=None):
    if scope is not None:
        with tf.variable_scope(scope):
            return sn_calc(w)
    shape = [s.value for s in w.shape]
    dimin = np.prod(shape[:-1])
    dimout = shape[-1]
    w = tf.reshape(w, (dimin, dimout))
    u = tf.get_variable(
        name='u',
        shape=(dimin, 1),
        initializer=tf.initializers.random_normal(0.5),
        dtype=tf.float32,
        trainable=False)

    ut, vt = power_iter(u, w)
    uup = tf.assign(u, ut, name='update_u')
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, uup)
    sn = spec_norm(uup, w)
    if len(shape) > 2:
        scale = np.sqrt(np.prod(shape[:-2]))
    else:
        scale = 1.
    # sn = tf.maximum(sn*scale, 1.)
    sn = sn * scale
    return sn


def sn_kernel(shape, scope, init=tf.initializers.glorot_normal()):
    with tf.variable_scope(scope) as vs:
        kernel_name = vs.name + "/kernel_sn:0"
        try:
            kernel = tf.get_default_graph().get_tensor_by_name(kernel_name)
            print("Existing kernel: {}".format(kernel_name))
            return kernel
        except KeyError:
            with tf.name_scope(vs.name + "/"):
                w = tf.get_variable(name='kernel_raw', shape=shape, initializer=init, dtype=tf.float32)
                sn = sn_calc(w)
                if len(shape) > 2:
                    # scale = tf.reduce_prod(tf.cast(tf.shape(w)[:-2], tf.float32))
                    # scale = tf.sqrt(tf.reduce_prod(tf.cast(tf.shape(w)[:-2], tf.float32)))
                    scale = 1.
                else:
                    scale = 1.
                kernel = tf.div(w, sn * scale, name='kernel_sn')
                s, u, v = tf.linalg.svd(tf.reshape(kernel, (-1, tf.shape(kernel)[-1])))
                tf.summary.scalar(vs.name + "/max_sv", tf.reduce_max(tf.abs(s)))
                print("New kernel: {}".format(kernel_name))
                return kernel


def sn_fully_connected(
        inputs,
        num_outputs,
        scope,
        activation_fn=tf.nn.relu,
weights_regularizer=None
):
    with tf.variable_scope(scope):
        input_dim = inputs.shape[-1].value
        assert input_dim
        kernel = sn_kernel(
            shape=(input_dim, num_outputs),
            scope='kernel'
        )
        bias = tf.get_variable(
            name='bias',
            shape=(num_outputs,),
            initializer=tf.initializers.zeros
        )
        rank = common_shapes.rank(inputs)
        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs = tf.tensordot(inputs, kernel, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            # if not context.executing_eagerly():
            #    shape = inputs.get_shape().as_list()
            #    output_shape = shape[:-1] + [self.units]
            #    outputs.set_shape(output_shape)
        else:
            # Cast the inputs to self.dtype, which is the variable dtype. We do not
            # cast if `should_cast_variables` is True, as in that case the variable
            # will be automatically casted to inputs.dtype.
            outputs = tf.mat_mul(inputs, kernel)
        outputs = tf.nn.bias_add(outputs, bias)
        if activation_fn is not None:
            return activation_fn(outputs)  # pylint: disable=not-callable
        else:
            return outputs
