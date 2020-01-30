import tensorflow as tf
from tensorflow.keras import layers
from model.synthesize.synthesize_factory import synthesizer_factory

import settings
import utils.util_funcs as uf
from utils.decorators import shape_check


class TotalLoss:
    def __init__(self, calc_losses):
        """
        :param calc_losses: list of (CalcLoss (class), weight)
        """
        self.calc_losses = calc_losses

    def __call__(self, predictions, features):
        """
        :param predictions: {"disp_ms": .., "pose": ..}
            disp_ms: multi scale disparity, list of [batch, height/scale, width/scale, 1]
            pose: 6-DoF poses [batch, num_src, 6]
        :param features: {"image": .., "pose_gt": .., "depth_gt": .., "intrinsic": ..}
            image: stacked image [batch, height*snippet_len, width, 3]
            intrinsic: camera projection matrix [batch, 3, 3]
        """
        augm_data = self.augment_data(features, predictions)

        losses = []
        for calc_loss, weight in self.calc_losses:
            loss = calc_loss(features, predictions, augm_data)
            losses.append(loss * weight)

        total_loss = layers.Lambda(lambda values: tf.reduce_sum(values, axis=0), name="total_loss")(losses)
        return total_loss

    def augment_data(self, features, predictions):
        augm_data = dict()
        pred_disp_ms = predictions['disp_ms']
        pred_pose = predictions['pose']
        pred_depth_ms = uf.disp_to_depth_tensor(pred_disp_ms)
        augm_data['depth_ms'] = pred_depth_ms

        stacked_image = features['image']
        intrinsic = features['intrinsic']
        source_image, target_image = uf.split_into_source_and_target(stacked_image)
        target_ms = uf.multi_scale_like(target_image, pred_disp_ms)
        augm_data['source'] = source_image
        augm_data['target'] = target_image
        augm_data['target_ms'] = target_ms

        return augm_data


class LossBase:
    def __call__(self, features, predictions, augm_data):
        raise NotImplementedError()


class PhotometricLossL1MultiScale(LossBase):
    def __call__(self, features, predictions, augm_data):
        """
        :return: photo_loss [batch]
        """
        pred_pose = predictions['pose']
        intrinsic = features['intrinsic']
        pred_depth_ms = augm_data['depth_ms']
        source_image = augm_data['source']
        original_target_ms = augm_data['target_ms']

        synth_target_ms = synthesizer_factory()(source_image, intrinsic, pred_depth_ms, pred_pose)

        losses = []
        for i, (synt_target, orig_target) in enumerate(zip(synth_target_ms, original_target_ms)):
            loss = layers.Lambda(lambda inputs: photometric_loss_l1(inputs[0], inputs[1]),
                                 name=f"photo_loss_{i}")([synt_target, orig_target])
            losses.append(loss)
        # sum over sources x scales, after stack over scales: [batch, num_src, num_scales]
        photo_loss = layers.Lambda(lambda data: tf.reduce_sum(tf.stack(data, axis=2), axis=[1, 2]),
                                   name="photo_loss_sum")(losses)
        return photo_loss


class SmoothenessLossMultiScale(LossBase):
    def __call__(self, features, predictions, augm_data):
        """
        :return: smootheness loss [batch]
        """
        pred_disp_ms = predictions['disp_ms']
        target_ms = augm_data['target_ms']
        losses = []
        orig_width = target_ms[0].get_shape().as_list()[2]
        for i, (disp, image) in enumerate(zip(pred_disp_ms, target_ms)):
            scale = orig_width / image.get_shape().as_list()[2]
            loss = layers.Lambda(lambda inputs: smootheness_loss(inputs[0], inputs[1]) / scale,
                                 name=f"smooth_loss_{i}")([disp, image])
            losses.append(loss)
        photo_loss = layers.Lambda(lambda data: tf.reduce_sum(tf.stack(data, axis=1), axis=1),
                                   name="smooth_loss_sum")(losses)
        return photo_loss


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

    # orig_target [batch, 1, height/scale, width/scale, 3]
    # axis=1 broadcasted in subtraction
    # photo_error: [batch, num_src, height/scale, width/scale, 3]
    photo_error = tf.abs(synt_target - orig_target)
    photo_error = tf.where(error_mask, tf.constant(0, dtype=tf.float32), photo_error)
    photo_loss = tf.reduce_mean(photo_error, axis=[2, 3, 4])
    return photo_loss


def smootheness_loss(disp, image):
    """
    :param disp: scaled disparity map, list of [batch, height/scale, width/scale, 1]
    :param image: scaled original target image [batch, height/scale, width/scale, 3]
    :return: smootheness loss [batch]
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
    smoothness = 0.5 * tf.reduce_mean(tf.abs(smoothness_x), axis=[1, 2, 3]) + \
                 0.5 * tf.reduce_mean(tf.abs(smoothness_y), axis=[1, 2, 3])
    return smoothness
