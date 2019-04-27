import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.gan.python.train import RunTrainOpsHook


def dis_train_hook(discriminator_scope:str, params):
    dis_losses = tf.losses.get_losses(scope=discriminator_scope)
    print("Discriminator losses: {}".format(dis_losses))
    dis_losses += tf.losses.get_regularization_losses(scope=discriminator_scope)
    dis_total_loss = tf.add_n(dis_losses)
    dis_optimizer = tf.train.AdamOptimizer(params.dis_lr)
    dis_updates = tf.get_collection(key=tf.GraphKeys.UPDATE_OPS, scope=discriminator_scope)
    dis_train_op = slim.learning.create_train_op(
        dis_total_loss,
        dis_optimizer,
        clip_gradient_norm=params.clip_gradient_norm,
        variables_to_train=tf.trainable_variables(scope=discriminator_scope),
        update_ops=dis_updates,
        global_step=None)
    discriminator_hook = RunTrainOpsHook(
        train_ops=[dis_train_op],
        train_steps=params.discriminator_steps
    )
    return discriminator_hook