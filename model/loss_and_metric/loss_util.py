import tensorflow as tf
from utils.decorators import shape_check


@shape_check
def photometric_loss_l1(synt_target, orig_target):
    """
    :param synt_target: scaled synthesized target image [batch, num_src, height/scale, width/scale, 3]
    :param orig_target: scaled original target image [batch, height/scale, width/scale, 3]
    :return: photo_loss [batch, num_src]
    """
    orig_target = tf.expand_dims(orig_target, axis=1)
    # create mask to ignore black region
    synt_target_gray = tf.reduce_mean(synt_target, axis=-1, keepdims=True)
    error_mask = tf.equal(synt_target_gray, 0)

    # orig_target: [batch, 1, height/scale, width/scale, 3]
    # axis=1 broadcasted in subtraction
    # photo_error: [batch, num_src, height/scale, width/scale, 3]
    photo_error = tf.abs(synt_target - orig_target)
    photo_error = tf.where(error_mask, tf.constant(0, dtype=tf.float32), photo_error)
    # average over image dimensions (h, w, c)
    photo_loss = tf.reduce_mean(photo_error)
    return photo_loss


@shape_check
def photometric_loss_l2(synt_target, orig_target):
    """
    :param synt_target: scaled synthesized target image [batch, num_src, height/scale, width/scale, 3]
    :param orig_target: scaled original target image [batch, height/scale, width/scale, 3]
    :return: photo_loss [batch, num_src]
    """
    orig_target = tf.expand_dims(orig_target, axis=1)
    # create mask to ignore black region
    synt_target_gray = tf.reduce_mean(synt_target, axis=-1, keepdims=True)
    error_mask = tf.equal(synt_target_gray, 0)

    # orig_target: [batch, 1, height/scale, width/scale, 3]
    # axis=1 broadcasted in subtraction
    # photo_error: [batch, num_src, height/scale, width/scale, 3]
    photo_error = tf.square(synt_target - orig_target)
    photo_error = tf.where(error_mask, tf.constant(0, dtype=tf.float32), photo_error)
    # average over image dimensions (h, w, c)
    photo_loss = tf.reduce_mean(photo_error)
    return photo_loss


@shape_check
def photometric_loss_ssim(synt_target, orig_target):
    """
    :param synt_target: scaled synthesized target image [batch, num_src, height/scale, width/scale, 3]
    :param orig_target: scaled original target image [batch, height/scale, width/scale, 3]
    :return: photo_loss [batch, num_src]
    """
    num_src = synt_target.get_shape().as_list()[1]
    orig_target = tf.expand_dims(orig_target, axis=1)
    orig_target = tf.tile(orig_target, [1, num_src, 1, 1, 1])
    # create mask to ignore black region
    synt_target_gray = tf.reduce_mean(synt_target, axis=-1, keepdims=True)
    error_mask = tf.equal(synt_target_gray, 0)

    x = orig_target     # [batch, num_src, height/scale, width/scale, 3]
    y = synt_target     # [batch, num_src, height/scale, width/scale, 3]
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ksize = [1, 3, 3]

    # TODO IMPORTANT!
    #   tf.nn.avg_pool results in error like ['NoneType' object has no attribute 'decode']
    #   when training model with gradient tape in eager mode,
    #   but no error in graph mode by @tf.function
    #   Instead, tf.keras.layers.AveragePooling3D results in NO error in BOTH modes
    # mu_x, mu_y: [batch, num_src, height/scale, width/scale, 3]
    average_pool = tf.keras.layers.AveragePooling3D(pool_size=ksize, strides=1, padding="SAME")
    mu_x = average_pool(x)
    mu_y = average_pool(y)
    # mu_x = tf.nn.avg_pool(x, ksize=ksize, strides=1, padding='SAME')
    # mu_y = tf.nn.avg_pool(y, ksize=ksize, strides=1, padding='SAME')

    sigma_x = average_pool(x ** 2) - mu_x ** 2
    sigma_y = average_pool(y ** 2) - mu_y ** 2
    sigma_xy = average_pool(x * y) - mu_x * mu_y

    ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    ssim = ssim_n / ssim_d
    ssim = tf.clip_by_value((1 - ssim) / 2, 0, 1)
    ssim = tf.where(error_mask, tf.constant(0, dtype=tf.float32), ssim)
    # average over image dimensions (h, w, c)
    ssim = tf.reduce_mean(ssim)
    return ssim
