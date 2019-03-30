import tensorflow as tf


def get_scale(params):
    scale = (
            (tf.cast(tf.train.get_or_create_global_step(), tf.float32) - params.anneal_start) /
            (params.anneal_end - params.anneal_start))
    scale = tf.clip_by_value(scale, params.anneal_min, 1)
    return scale


def get_kl_scale_logistic(params):
    return get_anneal_scale_logistic(
        anneal_start=params.kl_anneal_start,
        anneal_end=params.kl_anneal_end,
        anneal_min=params.kl_anneal_min,
        anneal_max=params.kl_anneal_max
    )


def get_penalty_scale_logistic(params):
    return get_anneal_scale_logistic(
        anneal_start=params.penalty_anneal_start,
        anneal_end=params.penalty_anneal_end,
        anneal_min=params.penalty_anneal_min,
        anneal_max=params.penalty_anneal_max
    )


def get_anneal_scale_logistic(anneal_start, anneal_end, anneal_min, anneal_max):
    if anneal_end > 0:
        mid = (anneal_end + anneal_start) / 2
        mult = (anneal_end - anneal_start) / 16
        scale = (tf.cast(tf.train.get_or_create_global_step(), tf.float32) - mid) / mult
        scale = tf.nn.sigmoid(scale)
        anneal_scale = anneal_max - anneal_min
        scale = (scale * anneal_scale) + anneal_min
        return scale
    else:
        return tf.constant(anneal_max, dtype=tf.float32, name='constant_scale')
