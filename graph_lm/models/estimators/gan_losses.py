import tensorflow as tf

from ...anneal import get_penalty_scale_logistic


def wgan_losses_v1(dis_out, n):
    dis_labels_real = tf.ones(shape=(n,), dtype=tf.float32)
    dis_labels_fake = -tf.ones(shape=(n,), dtype=tf.float32)
    dis_labels = tf.concat([dis_labels_real, dis_labels_fake], axis=0)
    wgan_loss_d_raw = tf.reduce_mean(dis_labels * dis_out)
    wgan_loss_g_raw = -wgan_loss_d_raw
    return wgan_loss_d_raw, wgan_loss_g_raw


def wgan_losses_v2(dis_out, n):
    dis_labels_real = tf.ones(shape=(n,), dtype=tf.float32)
    dis_labels_fake = -tf.ones(shape=(n,), dtype=tf.float32)
    dis_labels = tf.concat([dis_labels_real, dis_labels_fake], axis=0)
    wgan_loss_d_raw = tf.reduce_mean(dis_labels * dis_out)
    wgan_loss_g_raw = tf.square(wgan_loss_d_raw)
    return wgan_loss_d_raw, wgan_loss_g_raw


def wgan_losses_v3(dis_out, n):
    dis_labels_real = tf.ones(shape=(n,), dtype=tf.float32)
    dis_labels_fake = tf.zeros(shape=(n,), dtype=tf.float32)
    dis_labels = tf.concat([dis_labels_real, dis_labels_fake], axis=0)
    wgan_loss_d_raw = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=dis_labels,
        logits=dis_out
    )
    wgan_loss_d_raw = tf.reduce_mean(wgan_loss_d_raw)
    gen_labels_real = tf.zeros(shape=(n,), dtype=tf.float32)
    gen_labels_fake = tf.ones(shape=(n,), dtype=tf.float32)
    gen_labels = tf.concat([gen_labels_real, gen_labels_fake], axis=0)
    wgan_loss_g_raw = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=gen_labels,
        logits=dis_out
    )
    wgan_loss_g_raw = tf.reduce_mean(wgan_loss_g_raw)
    return wgan_loss_d_raw, wgan_loss_g_raw


def wgan_losses_v4(dis_out, n):
    dis_labels_real = tf.ones(shape=(n,), dtype=tf.float32)
    dis_labels_fake = tf.zeros(shape=(n,), dtype=tf.float32)
    dis_labels = tf.concat([dis_labels_real, dis_labels_fake], axis=0)
    wgan_loss_d_raw = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=dis_labels,
        logits=dis_out
    )
    wgan_loss_d_raw = tf.reduce_mean(wgan_loss_d_raw)
    pd = tf.nn.softplus(-dis_out)
    gen_labels_real = tf.zeros(shape=(n,), dtype=tf.float32)
    gen_labels_fake = tf.ones(shape=(n,), dtype=tf.float32)
    gen_labels = tf.concat([gen_labels_real, gen_labels_fake], axis=0)
    wgan_loss_g_raw = tf.reduce_sum(gen_labels * pd)/tf.cast(n, tf.float32)
    return wgan_loss_d_raw, wgan_loss_g_raw


def add_gan_losses(params, wgan_loss_d_raw, wgan_loss_g_raw, autoencoder_scope, discriminator_scope):
    wgan_scale = get_penalty_scale_logistic(params=params)
    wgan_loss_g = wgan_scale * wgan_loss_g_raw
    wgan_loss_d = wgan_scale * wgan_loss_d_raw
    tf.summary.scalar("wgan_loss_d_raw", wgan_loss_d_raw)
    tf.summary.scalar("wgan_loss_g_raw", wgan_loss_g_raw)
    tf.summary.scalar("wgan_scale", wgan_scale)
    tf.summary.scalar("wgan_loss_g", wgan_loss_g)
    tf.summary.scalar("wgan_loss_d", wgan_loss_d)
    with tf.name_scope(autoencoder_scope + "/"):
        tf.losses.add_loss(tf.identity(wgan_loss_g, name="wgan_loss_g"))
    with tf.name_scope(discriminator_scope + "/"):
        tf.losses.add_loss(tf.identity(wgan_loss_d, name="wgan_loss_d"))


def build_gan_losses(
        params, autoencoder_scope:str, discriminator_scope:str,
        dis_out, n
):
    with tf.name_scope("gan_losses"):
        if params.gan_loss == 'v1':
            wgan_loss_d_raw, wgan_loss_g_raw = wgan_losses_v1(dis_out=dis_out, n=n)
        elif params.gan_loss == 'v2':
            wgan_loss_d_raw, wgan_loss_g_raw = wgan_losses_v2(dis_out=dis_out, n=n)
        elif params.gan_loss == 'v3':
            wgan_loss_d_raw, wgan_loss_g_raw = wgan_losses_v3(dis_out=dis_out, n=n)
        elif params.gan_loss == 'v4':
            wgan_loss_d_raw, wgan_loss_g_raw = wgan_losses_v4(dis_out=dis_out, n=n)
        else:
            raise ValueError()

        add_gan_losses(
            params=params,
            wgan_loss_d_raw=wgan_loss_d_raw, wgan_loss_g_raw=wgan_loss_g_raw,
            autoencoder_scope=autoencoder_scope, discriminator_scope=discriminator_scope
        )
