import tensorflow as tf
from tensorflow.keras import layers


def photometric_loss_multi_scale(synthesized_target_ms, original_target_ms):
    """
    :param synthesized_target_ms: multi scale synthesized targets, list of
                                  [batch, num_src, height/scale, width/scale, 3]
    :param original_target_ms: multi scale target images, list of [batch, height, width, 3]
    :return: photo_loss scalar
    """
    losses = []
    for i, (synt_target, orig_target) in enumerate(zip(synthesized_target_ms, original_target_ms)):
        loss = layers.Lambda(lambda inputs: photometric_loss(inputs[0], inputs[1]),
                             name=f"photo_loss_{i}")([synt_target, orig_target])
        losses.append(loss)
    photo_loss = layers.Lambda(lambda data: tf.reduce_sum(tf.stack(data, axis=1), axis=1),
                               name="photo_loss_sum")(losses)
    return photo_loss


def photometric_loss(synt_target, orig_target):
    """
    :param synt_target: scaled synthesized target image [batch, num_src, height/scale, width/scale, 3]
    :param orig_target: scaled original target image [batch, height/scale, width/scale, 3]
    :return: scalar loss
    """
    orig_target = tf.expand_dims(orig_target, axis=1)
    # create mask to ignore black region
    synt_target_gray = tf.reduce_mean(synt_target, axis=-1, keepdims=True)
    error_mask = tf.equal(synt_target_gray, 0)

    # orig_target [batch, 1, height/scale, width/scale, 3]
    # axis=1 broadcasted in subtraction
    photo_error = tf.abs(synt_target - orig_target)
    photo_error = tf.where(error_mask, tf.constant(0, dtype=tf.float32), photo_error)
    # photo_error: [batch, num_src, height/scale, width/scale, 3]
    photo_loss = tf.reduce_sum(tf.reduce_mean(photo_error, axis=[2, 3, 4]), axis=1)
    return photo_loss


def smootheness_loss_multi_scale(disp_ms, image_ms, height_orig):
    """
    :param disp_ms: multi scale disparity map, list of [batch, height/scale, width/scale, 1]
    :param image_ms: multi scale image, list of [batch, height/scale, width/scale, 3]
    :return: photometric loss (scalar)
    """
    losses = []
    for i, (disp, image) in enumerate(zip(disp_ms, image_ms)):
        scale = height_orig // image_ms.get_shape().as_list()
        loss = layers.Lambda(lambda inputs: smootheness_loss(inputs[0], inputs[1]) / scale,
                             name=f"smooth_loss_{i}")([disp, image])
        losses.append(loss)
    photo_loss = layers.Lambda(lambda data: tf.reduce_sum(tf.stack(data, axis=1), axis=1),
                               name="smooth_loss_sum")(losses)
    return photo_loss


def smootheness_loss(disp, image):
    """
    :param disp: scaled disparity map, list of [batch, height/scale, width/scale, 1]
    :param image: scaled original target image [batch, height/scale, width/scale, 3]
    :return: smootheness loss (scalar)
    """
    def gradient_x(img):
        gx = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gx

    def gradient_y(img):
        gy = img[:, :-1, :, :] - img[:, 1:, :, :]
        return gy

    disp_gradients_x = gradient_x(disp)
    disp_gradients_y = gradient_y(disp)

    image_gradients_x = gradient_x(image)
    image_gradients_y = gradient_y(image)

    weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x), 3, keepdims=True))
    weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y), 3, keepdims=True))

    # [batch, height/scale, width/scale, 1]
    smoothness_x = disp_gradients_x * weights_x
    smoothness_y = disp_gradients_y * weights_y

    # return [batch]
    smoothness = 0.5*tf.reduce_mean(tf.abs(smoothness_x), axis=[1, 2, 3]) + \
                 0.5*tf.reduce_mean(tf.abs(smoothness_y), axis=[1, 2, 3])
    return smoothness


def depth_error_metric(depth_pred, depth_true):
    """
    :param depth_pred: predicted depth [batch, height, width, 1]
    :param depth_true: ground truth depth [batch, height, width, 1]
    :return: depth error metric (scalar)
    """
    # flatten depths
    batch, height, width, _ = depth_pred.get_shape().as_list()
    depth_pred_vec = tf.reshape(depth_pred, (batch, height*width))
    depth_true_vec = tf.reshape(depth_true, (batch, height*width))

    # filter out zero depths
    depth_invalid_mask = tf.math.equal(depth_true_vec, 0)
    depth_pred_vec = tf.where(depth_invalid_mask, tf.constant(0, dtype=tf.float32), depth_pred_vec)
    depth_true_vec = tf.where(depth_invalid_mask, tf.constant(0, dtype=tf.float32), depth_true_vec)

    # normalize depths, [height*width, batch] / [batch] = [height*width, batch]
    depth_pred_vec = tf.transpose(depth_pred_vec) / tf.reduce_mean(depth_pred_vec, axis=1)
    depth_true_vec = tf.transpose(depth_true_vec) / tf.reduce_mean(depth_true_vec, axis=1)
    # [height*width, batch] -> [batch]
    depth_error = tf.reduce_mean(tf.abs(depth_pred_vec - depth_true_vec), axis=0)
    return depth_error


# ==================== tests ====================
import numpy as np


def test_depth_error_metric():
    depth_pred = np.zeros((8, 10, 10, 1))
    depth_pred[:, :3] = 1.6
    depth_pred[:, 3:6] = 2.4
    print("depth pred\n", depth_pred[0, :, :, 0])
    depth_pred = tf.constant(depth_pred, dtype=tf.float32)
    depth_true = tf.constant(np.ones((8, 10, 10, 1)), dtype=tf.float32)

    error = depth_error_metric(depth_true, depth_pred)
    assert np.isclose(error, 0.2).all()

    print("!!! test_depth_error_metric passed")


def test():
    test_depth_error_metric()


if __name__ == "__main__":
    test()
