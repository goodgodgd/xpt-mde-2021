import tensorflow as tf
from tensorflow.keras import layers
from model.synthesize.synthesize_base import SynthesizeMultiScale

import utils.util_funcs as uf
from utils.util_class import WrongInputException
import utils.convert_pose as cp
from utils.decorators import shape_check


class TotalLoss:
    def __init__(self, calc_losses=None, weights=None, stereo=False):
        """
        :param calc_losses: list of loss calculators
        :param weights: list of weights of loss calculators
        """
        self.calc_losses = calc_losses
        self.weights = weights
        self.stereo = stereo

    @shape_check
    def __call__(self, predictions, features):
        """
        :param predictions: {"disp_ms": .., "pose": ..}
            disp_ms: multi scale disparity, list of [batch, height/scale, width/scale, 1]
            pose: 6-DoF poses [batch, num_src, 6]
        :param features: {"image": .., "pose_gt": .., "depth_gt": .., "intrinsic": ..}
            image: stacked image [batch, height*snippet_len, width, 3]
            intrinsic: camera projection matrix [batch, 3, 3]
        :return loss_batch: final loss of frames in batch [batch]
                losses: list of losses computed from calc_losses
        """
        augm_data = self.augment_data(features, predictions)
        if self.stereo:
            augm_data_rig = self.augment_data(features, predictions, "_R")
            augm_data.update(augm_data_rig)

        losses = []
        for calc_loss, weight in zip(self.calc_losses, self.weights):
            loss = calc_loss(features, predictions, augm_data)
            losses.append(loss * weight)

        loss_batch = layers.Lambda(lambda values: tf.reduce_sum(values, axis=0), name="losses")(losses)
        return loss_batch, losses

    def augment_data(self, features, predictions, suffix=""):
        """
        gather additional data required to compute losses
        :param features: {image, intrinsic}
                image: stacked image snippet [batch, snippet_len*height, width, 3]
                intrinsic: camera projection matrix [batch, 3, 3]
        :param predictions: {disp_ms, pose}
                disp_ms: multi scale disparities, list of [batch, height/scale, width/scale, 1]
                pose: poses that transform points from target to source [batch, num_src, 6]
        :param suffix: suffix to keys
        :return augm_data: {depth_ms, source, target, target_ms, synth_target_ms}
                depth_ms: multi scale depth, list of [batch, height/scale, width/scale, 1]
                source: source frames [batch, num_src*height, width, 3]
                target: target frame [batch, height, width, 3]
                target_ms: multi scale target frame, list of [batch, height/scale, width/scale, 3]
                synth_target_ms: multi scale synthesized target frames generated from each source image,
                                list of [batch, num_src, height/scale, width/scale, 3]
        """
        augm_data = dict()
        pred_disp_ms = predictions["disp_ms" + suffix]
        pred_pose = predictions["pose" + suffix]
        pred_depth_ms = uf.disp_to_depth_tensor(pred_disp_ms)
        augm_data["depth_ms" + suffix] = pred_depth_ms

        stacked_image = features["image" + suffix]
        intrinsic = features["intrinsic" + suffix]
        source_image, target_image = uf.split_into_source_and_target(stacked_image)
        target_ms = uf.multi_scale_like(target_image, pred_disp_ms)
        augm_data["source" + suffix] = source_image
        augm_data["target" + suffix] = target_image
        augm_data["target_ms" + suffix] = target_ms

        synth_target_ms = SynthesizeMultiScale()(source_image, intrinsic, pred_depth_ms, pred_pose)
        augm_data["synth_target_ms" + suffix] = synth_target_ms

        return augm_data


class LossBase:
    def __call__(self, features, predictions, augm_data):
        raise NotImplementedError()


class PhotometricLossMultiScale(LossBase):
    def __init__(self, method, key_suffix=""):
        if method == "L1":
            self.photometric_loss = photometric_loss_l1
        elif method == "SSIM":
            self.photometric_loss = photometric_loss_ssim
        else:
            raise WrongInputException("Wrong photometric loss name: " + method)

        self.key_suffix = key_suffix

    def __call__(self, features, predictions, augm_data):
        """
        desciptions of inputs are available in 'TotalLoss.augment_data()'
        :return: photo_loss [batch]
        """
        original_target_ms = augm_data["target_ms" + self.key_suffix]
        synth_target_ms = augm_data["synth_target_ms" + self.key_suffix]

        losses = []
        for i, (synt_target, orig_target) in enumerate(zip(synth_target_ms, original_target_ms)):
            loss = layers.Lambda(lambda inputs: self.photometric_loss(inputs[0], inputs[1]),
                                 name=f"photo_loss_{i}")([synt_target, orig_target])
            losses.append(loss)
        # sum over sources x scales, after stack over scales: [batch, num_src, num_scales]
        batch_loss = layers.Lambda(lambda data: tf.reduce_sum(tf.stack(data, axis=2), axis=[1, 2]),
                                   name="photo_loss_sum")(losses)
        return batch_loss


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
    photo_loss = tf.reduce_mean(photo_error, axis=[2, 3, 4])
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
    # mu_x, mu_y: [batch, num_src, height/scale, width/scale, 3]
    mu_x = tf.nn.avg_pool(x, ksize=ksize, strides=1, padding='SAME')
    mu_y = tf.nn.avg_pool(y, ksize=ksize, strides=1, padding='SAME')

    sigma_x = tf.nn.avg_pool(x ** 2, ksize, 1, 'SAME') - mu_x ** 2
    sigma_y = tf.nn.avg_pool(y ** 2, ksize, 1, 'SAME') - mu_y ** 2
    sigma_xy = tf.nn.avg_pool(x * y, ksize, 1, 'SAME') - mu_x * mu_y

    ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    ssim = ssim_n / ssim_d
    ssim = tf.clip_by_value((1 - ssim) / 2, 0, 1)
    ssim = tf.where(error_mask, tf.constant(0, dtype=tf.float32), ssim)
    # average over image dimensions (h, w, c)
    ssim = tf.reduce_mean(ssim, axis=[2, 3, 4])
    return ssim


class SmoothenessLossMultiScale(LossBase):
    def __init__(self, key_suffix=""):
        self.key_suffix = key_suffix

    def __call__(self, features, predictions, augm_data):
        """
        desciptions of inputs are available in "PhotometricLossL1MultiScale"
        :return: smootheness loss [batch]
        """
        pred_disp_ms = predictions["disp_ms" + self.key_suffix]
        target_ms = augm_data["target_ms" + self.key_suffix]
        losses = []
        orig_width = target_ms[0].get_shape().as_list()[2]
        for i, (disp, image) in enumerate(zip(pred_disp_ms, target_ms)):
            scale = orig_width / image.get_shape().as_list()[2]
            loss = layers.Lambda(lambda inputs: self.smootheness_loss(inputs[0], inputs[1]) / scale,
                                 name=f"smooth_loss_{i}")([disp, image])
            losses.append(loss)

        batch_loss = layers.Lambda(lambda x: tf.reduce_sum(tf.stack(x, axis=1), axis=1),
                                   name="smooth_loss_sum")(losses)
        return batch_loss

    def smootheness_loss(self, disp, image):
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

        smoothness_x = 0.5 * tf.reduce_mean(tf.abs(smoothness_x), axis=[1, 2, 3])
        smoothness_y = 0.5 * tf.reduce_mean(tf.abs(smoothness_y), axis=[1, 2, 3])
        smoothness = smoothness_x + smoothness_y
        return smoothness


class StereoDepthLoss(LossBase):
    def __init__(self, method):
        if method == "L1":
            self.photometric_loss = photometric_loss_l1
        elif method == "SSIM":
            self.photometric_loss = photometric_loss_ssim
        else:
            raise WrongInputException("Wrong photometric loss name: " + method)

    def __call__(self, features, predictions, augm_data):
        """
        desciptions of inputs are available in 'TotalLoss.augment_data()'
        :return: photo_loss [batch]
        """
        # synthesize left image from right image
        loss_left, _ = self.stereo_synthesize_loss(source_img=augm_data["target_R"],
                                                   target_ms=augm_data["target_ms"],
                                                   disp_tgt_ms=predictions["disp_ms"],
                                                   pose_t2s=tf.linalg.inv(features["stereo_T_LR"]),
                                                   intrinsic=features["intrinsic"])
        loss_right, _ = self.stereo_synthesize_loss(source_img=augm_data["target"],
                                                    target_ms=augm_data["target_ms_R"],
                                                    disp_tgt_ms=predictions["disp_ms_R"],
                                                    pose_t2s=features["stereo_T_LR"],
                                                    intrinsic=features["intrinsic_R"],
                                                    suffix="_R")
        losses = loss_left + loss_right

        # sum over sources x scales, after stack over scales: [batch, num_src, num_scales]
        batch_loss = layers.Lambda(lambda data: tf.reduce_sum(tf.stack(data, axis=2), axis=[1, 2]),
                                   name="photo_loss_sum")(losses)
        return batch_loss

    def stereo_synthesize_loss(self, source_img, target_ms, disp_tgt_ms, pose_t2s, intrinsic, suffix=""):
        """
        synthesize image from source to target
        :param source_img: [batch, num_src*height, width, 3]
        :param target_ms: [batch, height, width, 3]
        :param disp_tgt_ms: [batch, height, width, 1]
        :param pose_t2s: [batch, num_src, 4, 4]
        :param intrinsic: [batch, num_src, 3, 3]
        :param suffix: "" if right to left, else "_R"
        """
        depth_ms = uf.disp_to_depth_tensor(disp_tgt_ms)
        pose_stereo = cp.pose_matr2rvec_batch(tf.expand_dims(pose_t2s, 1))

        # synth_xxx_ms: list of [batch, 1, height/scale, width/scale, 3]
        synth_target_ms = SynthesizeMultiScale()(source_img, intrinsic, depth_ms, pose_stereo)
        losses = []
        for i, (synth_img_sc, target_img_sc, depth) in enumerate(zip(synth_target_ms, target_ms, depth_ms)):
            loss = layers.Lambda(lambda inputs: self.photometric_loss(inputs[0], inputs[1]),
                                 name=f"photo_loss_{i}" + suffix)([synth_img_sc, target_img_sc])
            losses.append(loss)
        return losses, synth_target_ms
