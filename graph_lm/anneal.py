import tensorflow as tf


def get_scale(params):
    scale = (
            (tf.cast(tf.train.get_or_create_global_step(), tf.float32) - params.anneal_start) /
            (params.anneal_end - params.anneal_start))
    scale = tf.clip_by_value(scale, params.anneal_min, 1)
    tf.summary.scalar("anneal_scale", scale)
    return scale


def get_scale_logistic(params):
    mid = (params.anneal_end + params.anneal_start) / 2
    mult = (params.anneal_end - params.anneal_start) / 20
    scale = (tf.cast(tf.train.get_or_create_global_step(), tf.float32) - mid) / mult
    scale = tf.nn.sigmoid(scale)
    scale = tf.clip_by_value(scale, params.anneal_min, 1)
    tf.summary.scalar("anneal_scale", scale)
    return scale
