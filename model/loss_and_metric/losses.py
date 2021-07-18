import tensorflow as tf
from tensorflow.keras import layers
from model.synthesize.synthesize_base import SynthesizeMultiScale
from model.synthesize.flow_warping import FlowWarpMultiScale

import utils.util_funcs as uf
from utils.util_class import WrongInputException
import utils.convert_pose as cp
from utils.decorators import shape_check
import model.loss_and_metric.loss_util as lsu
from config import opts


class TotalLoss:
    def __init__(self, loss_objects=None, loss_weights=None, stereo=False, batch_size=1):
        """
        :param loss_objects: dict of loss objects
        :param loss_weights: dict of weights of losses
        """
        self.loss_objects = loss_objects
        self.loss_weights = loss_weights
        self.stereo = stereo
        self.batch_size = batch_size

    @shape_check
    def __call__(self, predictions, features):
        """
        :param predictions: {"depth_ms": .., "disp_ms": .., "pose": ..}
            disp_ms: multi scale disparity, list of [batch, height/scale, width/scale, 1]
            pose: 6-DoF poses [batch, numsrc, 6]
        :param features: {"image": .., "pose_gt": .., "depth_gt": .., "intrinsic": ..}
            image: stacked image [batch, height*snippet_len, width, 3]
            intrinsic: camera projection matrix [batch, 3, 3]
        :return loss: final loss of frames in batch (scalar)
                losses: list of losses computed from loss_objects
        """
        augm_data = self.append_data(features, predictions)
        if self.stereo and ("image_R" in features):
            augm_data_rig = self.append_data(features, predictions, "_R")
            augm_data.update(augm_data_rig)
            augm_data_stereo = self.synethesize_stereo(features, predictions, augm_data)
            augm_data.update(augm_data_stereo)

        losses = []
        loss_by_type = dict()
        for loss_name in self.loss_objects:
            loss_batch = self.loss_objects[loss_name](features, predictions, augm_data)
            # [batch] -> scalar
            loss_mean = tf.nn.compute_average_loss(loss_batch, global_batch_size=self.batch_size)
            weighted_loss = loss_mean * self.loss_weights[loss_name]
            losses.append(weighted_loss)
            loss_by_type[loss_name] = loss_mean

        total_loss = layers.Lambda(lambda x: tf.reduce_sum(x), name="total_loss")(losses)
        return total_loss, loss_by_type

    def append_data(self, features, predictions, suffix=""):
        """
        gather additional data required to compute losses
        :param features: {image, intrinsic}
                image: stacked image snippet [batch, snippet_len*height, width, 3]
                intrinsic: camera projection matrix [batch, 3, 3]
        :param predictions: {depth_ms, pose}
                depth_ms: multi scale disparities, list of [batch, height/scale, width/scale, 1]
                pose: poses that transform points from target to source [batch, numsrc, 6]
        :param suffix: suffix to keys
        :return augm_data: {depth_ms, source, target, target_ms, synth_target_ms}
                depth_ms: multi scale depth, list of [batch, height/scale, width/scale, 1]
                source: source frames [batch, numsrc, height, width, 3]
                target: target frame [batch, height, width, 3]
                target_ms: multi scale target frame, list of [batch, height/scale, width/scale, 3]
                synth_target_ms: multi scale synthesized target frames generated from each source image,
                                list of [batch, numsrc, height/scale, width/scale, 3]
                warped_target_ms: multi scale flow warped target frames generated from each source image,
                                list of [batch, numsrc, height/scale, width/scale, 3]
        """
        image5d = features["image5d" + suffix]
        intrinsic = features["intrinsic" + suffix]
        source_image = image5d[:, :-1]
        target_image = image5d[:, -1]
        augm_data = dict()
        augm_data["source" + suffix] = source_image
        augm_data["target" + suffix] = target_image

        if ("depth_ms" + suffix in predictions) and ("pose" + suffix in predictions):
            pred_depth_ms = predictions["depth_ms" + suffix]
            pred_pose = predictions["pose" + suffix]
            target_ms = uf.multi_scale_like_depth(target_image, pred_depth_ms)
            augm_data["target_ms" + suffix] = target_ms
            # synthesized image is used in both L1 and SSIM photometric losses
            synth_target_ms = SynthesizeMultiScale()(source_image, intrinsic, pred_depth_ms, pred_pose)
            augm_data["synth_target_ms" + suffix] = synth_target_ms

        # warped image is used in both L1 and SSIM photometric losses
        if "flow_ms" + suffix in predictions:
            pred_flow_ms = predictions["flow_ms" + suffix]
            # flows have lower resolution than depths, so "target_ms" is not appropriate for flows
            flow_target_ms = uf.multi_scale_like_flow(target_image, pred_flow_ms)
            augm_data["flow_target_ms" + suffix] = flow_target_ms
            warped_target_ms = FlowWarpMultiScale()(source_image, pred_flow_ms)
            augm_data["warped_target_ms" + suffix] = warped_target_ms

        return augm_data

    def synethesize_stereo(self, features, predictions, augm_data):
        """
        gather additional data required to compute losses
        :param features: {image, intrinsic}
                intrinsic: camera projection matrix [batch, 3, 3]
        :param predictions: {depth_ms, stereo_T_LR}
                depth_ms: multi scale disparities, list of [batch, height/scale, width/scale, 1]
                stereo_T_LR: poses that transform points from right to left [batch, numsrc, 6]
        :param augm_data: {source, target, target_ms, synth_target_ms}
                target: target frame [batch, height, width, 3]
        """
        synth_stereo = dict()
        if ("stereo_T_LR" not in features) or ("depth_ms" not in predictions):
            return synth_stereo

        # synthesize left image from right image
        pose_T_RL = tf.linalg.inv(features["stereo_T_LR"])
        pose_T_RL = cp.pose_matr2rvec_batch(tf.expand_dims(pose_T_RL, 1))
        stereo_source = tf.expand_dims(augm_data["target_R"], 1)
        synth_stereo["stereo_synth_ms"] = SynthesizeMultiScale()(
                                                source_image=stereo_source,
                                                intrinsic=features["intrinsic"],
                                                pred_depth_ms=predictions["depth_ms"],
                                                pred_pose=pose_T_RL)

        # synthesize right image from left image
        pose_T_LR = features["stereo_T_LR"]
        pose_T_LR = cp.pose_matr2rvec_batch(tf.expand_dims(pose_T_LR, 1))
        stereo_source = tf.expand_dims(augm_data["target"], 1)
        synth_stereo["stereo_synth_ms_R"] = SynthesizeMultiScale()(
                                                source_image=stereo_source,
                                                intrinsic=features["intrinsic"],
                                                pred_depth_ms=predictions["depth_ms_R"],
                                                pred_pose=pose_T_LR)
        # synth_stereo_xxx: list of [batch, 1, height/scale, width/scale, 3]
        return synth_stereo


class LossBase:
    def __call__(self, features, predictions, augm_data):
        raise NotImplementedError()

    def merge_multi_scale_losses(self, losses, name):
        """
        :param losses: list of tensors [scales, batch]
        :return: [batch]
        """
        # [scales, batch] -> transpose: [batch, scales] -> matmul: [batch]
        loss_batch = layers.Lambda(lambda x: tf.matmul(tf.transpose(x), self.scale_weights), name=name)(losses)
        return loss_batch


class PhotometricLoss(LossBase):
    def __init__(self, method, scale_weights, key_suffix=""):
        if method == "L1":
            self.photometric_loss = lsu.photometric_loss_l1
        elif method == "L2":
            self.photometric_loss = lsu.photometric_loss_l2
        elif method == "SSIM":
            self.photometric_loss = lsu.photometric_loss_ssim
        else:
            raise WrongInputException("Wrong photometric loss name: " + method)

        self.key_suffix = key_suffix
        self.scale_weights = scale_weights

    def __call__(self, features, predictions, augm_data):
        raise NotImplementedError()


class PhotometricLossMultiScale(PhotometricLoss):
    def __init__(self, method, scale_weights, key_suffix=""):
        super().__init__(method, scale_weights, key_suffix)

    def __call__(self, features, predictions, augm_data):
        """
        desciptions of inputs are available in 'TotalLoss.append_data()'
        :return: photo_loss [batch]
        """
        original_target_ms = augm_data["target_ms" + self.key_suffix]
        synth_target_ms = augm_data["synth_target_ms" + self.key_suffix]

        losses = []
        for i, (synt_target, orig_target) in enumerate(zip(synth_target_ms, original_target_ms)):
            loss = layers.Lambda(lambda inputs: self.photometric_loss(inputs[0], inputs[1]),
                                 name=f"photo_loss_{i}" + self.key_suffix)([synt_target, orig_target])
            losses.append(loss)

        name = "photo_loss_sum" + self.key_suffix
        # weighted sum over scales: [scales, batch] -> [batch]
        return self.merge_multi_scale_losses(losses, name)


class MonoDepth2LossMultiScale(PhotometricLoss):
    """
    compare photo losses of source images and take only min of source losses for each pixel
    """
    def __init__(self, method, scale_weights, key_suffix=""):
        super().__init__(method, scale_weights, key_suffix)

    def __call__(self, features, predictions, augm_data):
        """
        desciptions of inputs are available in 'TotalLoss.append_data()'
        :return: photo_loss [batch]
        """
        synth_target_ms = augm_data["synth_target_ms" + self.key_suffix]
        original_target = augm_data["target" + self.key_suffix]
        Ho, Wo = original_target.shape[1:3]

        losses = []
        for i, synt_target in enumerate(synth_target_ms):
            # resize synthesized target to original image size
            synt_target_rsz = resize_bilinear(synt_target, (Ho, Wo))
            op_name = f"mono2_loss_{i}" + self.key_suffix
            # compute L1 loss without reduce_mean -> [batch, numsrc, height, width, channel]
            loss = layers.Lambda(lambda inputs: self.photometric_loss(inputs[0], inputs[1], False),
                                 name=op_name)([synt_target_rsz, original_target])
            # take minimum loss over sources for each pixel -> [batch, height, width, channel]
            op_name = f"mono2_loss_min_{i}" + self.key_suffix
            loss = layers.Lambda(lambda x: tf.reduce_min(x, axis=1), name=op_name)(loss)
            losses.append(loss)

        # average over image [scales, batch, height, width, channel] -> [scales, batch]
        op_name = "mono2_loss_mean" + self.key_suffix
        losses = layers.Lambda(lambda x: tf.reduce_mean(x, axis=[2, 3, 4]), name=op_name)(losses)
        op_name = "mono2_loss_sum" + self.key_suffix
        # weighted sum over scales: [scales, batch] -> [batch]
        return self.merge_multi_scale_losses(losses, op_name)


class CombinedLossMultiScale(PhotometricLoss):
    """
    mask static flow loss where static flow < optical flow, and average masked static loss
    """
    def __init__(self, method, scale_weights, key_suffix=""):
        super().__init__(method, scale_weights, key_suffix)

    def __call__(self, features, predictions, augm_data):
        """
        desciptions of inputs are available in 'TotalLoss.append_data()'
        :return: photo_loss [batch]
        """
        synth_target_ms = augm_data["synth_target_ms" + self.key_suffix]
        warped_target_ms = augm_data["warped_target_ms" + self.key_suffix]
        original_target = augm_data["target" + self.key_suffix]
        Ho, Wo = original_target.shape[1:3]

        # compare multi scale static flow with fixed scale optical flow with 1/4 scale
        warped_target = warped_target_ms[0]
        warped_target = resize_bilinear(warped_target, (Ho, Wo))
        op_name = f"comb_flow_loss_{0}" + self.key_suffix
        # flow loss: [batch, numsrc, height, width, 3]
        flow_loss = layers.Lambda(lambda inputs: self.photometric_loss(inputs[0], inputs[1], False),
                                  name=op_name)([warped_target, original_target])

        losses = []
        for i, synt_target in enumerate(synth_target_ms):
            # resize synthesized target to original image size
            synt_target_rsz = resize_bilinear(synt_target, (Ho, Wo))
            op_name = f"comb_mono2_loss_{i}" + self.key_suffix
            # compute L1 loss without reduce_mean -> [batch, numsrc, height, width, 3]
            static_loss = layers.Lambda(lambda inputs: self.photometric_loss(inputs[0], inputs[1], False),
                                        name=op_name)([synt_target_rsz, original_target])

            # extract static loss lower than optical flow loss
            mask = tf.cast(static_loss < flow_loss, tf.float32)
            static_loss = static_loss * mask
            # reduce mean -> [batch]
            op_name = f"comb_photo_mean_{i}" + self.key_suffix
            loss = layers.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2, 3, 4]), name=op_name)(static_loss)
            losses.append(loss)

        op_name = f"comb_photo_sum" + self.key_suffix
        # weighted sum over scales: [scales, batch] -> [batch]
        return self.merge_multi_scale_losses(losses, op_name)


class MoALossMultiScale(PhotometricLoss):
    """
    minimum over temporal photometric losses and stereo photometric loss
    """
    def __init__(self, method, scale_weights, key_suffix=""):
        super().__init__(method, scale_weights, key_suffix)

    def __call__(self, features, predictions, augm_data):
        """
        desciptions of inputs are available in 'TotalLoss.append_data()'
        :return: photo_loss [batch]
        """
        temp_synth_target_ms = augm_data["synth_target_ms" + self.key_suffix]
        stro_synth_target_ms = augm_data["stereo_synth_ms"]
        original_target = augm_data["target" + self.key_suffix]
        Ho, Wo = original_target.shape[1:3]

        losses = []
        for i, (temp_target, stro_target) in enumerate(zip(temp_synth_target_ms, stro_synth_target_ms)):
            # resize synthesized target to original image size
            temp_target_rs = resize_bilinear(temp_target, (Ho, Wo))
            stro_target_rs = resize_bilinear(stro_target, (Ho, Wo))

            # photometric loss: [batch, numsrc, height, width, 3]
            op_name = f"moa_temp_loss_{i}" + self.key_suffix
            temp_loss = layers.Lambda(lambda inputs: self.photometric_loss(inputs[0], inputs[1], False),
                                        name=op_name)([temp_target_rs, original_target])
            op_name = f"moa_stro_loss_{i}" + self.key_suffix
            stro_loss = layers.Lambda(lambda inputs: self.photometric_loss(inputs[0], inputs[1], False),
                                      name=op_name)([stro_target_rs, original_target])
            # moa loss: [batch, numsrc*2, height, width, 3]
            moa_loss = tf.concat([temp_loss, stro_loss], axis=1, name=f"moa_cat_{i}" + self.key_suffix)
            # moa loss: [batch, height, width, 3]
            moa_loss = tf.reduce_min(moa_loss, axis=1, name=f"moa_loss_{i}" + self.key_suffix)
            moa_loss = tf.reduce_mean(moa_loss, axis=[1, 2, 3], name=f"moa_loss_mean_{i}" + self.key_suffix)
            losses.append(moa_loss)

        op_name = f"comb_photo_sum" + self.key_suffix
        # weighted sum over scales: [scales, batch] -> [batch]
        return self.merge_multi_scale_losses(losses, op_name)


class MD2CombLossMultiScale(PhotometricLoss):
    """
    mask static flow loss where static flow < optical flow, and average masked static loss
    """
    def __init__(self, method, scale_weights, key_suffix=""):
        super().__init__(method, scale_weights, key_suffix)

    def __call__(self, features, predictions, augm_data):
        """
        desciptions of inputs are available in 'TotalLoss.append_data()'
        :return: photo_loss [batch]
        """
        synth_target_ms = augm_data["synth_target_ms" + self.key_suffix]
        warped_target_ms = augm_data["warped_target_ms" + self.key_suffix]
        original_target = augm_data["target" + self.key_suffix]
        Ho, Wo = original_target.shape[1:3]

        # compare multi scale static flow with fixed scale optical flow with 1/4 scale
        warped_target = warped_target_ms[0]
        warped_target = resize_bilinear(warped_target, (Ho, Wo))
        op_name = f"comb_flow_loss_{0}" + self.key_suffix
        # flow loss: [batch, numsrc, height, width, 3]
        flow_loss = layers.Lambda(lambda inputs: self.photometric_loss(inputs[0], inputs[1], False),
                                  name=op_name)([warped_target, original_target])

        losses = []
        for i, synt_target in enumerate(synth_target_ms):
            # resize synthesized target to original image size
            synt_target_rsz = resize_bilinear(synt_target, (Ho, Wo))
            op_name = f"md2comb_loss_{i}" + self.key_suffix
            # compute L1 loss without reduce_mean [batch, numsrc, height, width, channel]
            static_loss = layers.Lambda(lambda inputs: self.photometric_loss(inputs[0], inputs[1], False),
                                        name=op_name)([synt_target_rsz, original_target])

            # add large value where static loss is larger than flow loss [batch, numsrc, height, width, channel]
            mask = tf.cast(static_loss > flow_loss * 2., tf.float32)
            static_loss = static_loss + mask * 1000.

            # take minimum loss over sources for each pixel -> [batch, height, width, channel]
            op_name = f"md2comb_loss_min_{i}" + self.key_suffix
            static_loss = layers.Lambda(lambda x: tf.reduce_min(x, axis=1), name=op_name)(static_loss)

            # average static over non-large values -> loss: [batch]
            mask = tf.cast(static_loss < 1000., tf.float32)
            op_name = f"md2comb_photo_sum_{i}" + self.key_suffix
            loss = tf.reduce_sum(static_loss * mask, axis=[1, 2, 3], name=op_name) / tf.math.count_nonzero(mask)
            losses.append(loss)

        op_name = f"comb_photo_sum" + self.key_suffix
        # weighted sum over scales: [scales, batch] -> [batch]
        return self.merge_multi_scale_losses(losses, op_name)


def resize_bilinear(srcimg, dst_hw):
    Hd, Wd = dst_hw
    B, N, Hs, Ws, C = srcimg.get_shape()
    srcimg = tf.reshape(srcimg, (B * N, Hs, Ws, C))
    dstimg = tf.image.resize(srcimg, (Hd, Wd), method="bilinear")
    dstimg = tf.reshape(dstimg, (B, N, Hd, Wd, C))
    return dstimg


class SmoothenessLossMultiScale(LossBase):
    def __init__(self, scale_weights, key_suffix=""):
        self.key_suffix = key_suffix
        self.scale_weights = scale_weights

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

        # weighted sum over scales: [scales, batch] -> [batch]
        return self.merge_multi_scale_losses(losses, "smooth_loss_sum")

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

        grad_factor = opts.IMAGE_GRADIENT_FACTOR
        weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x * grad_factor), 3, keepdims=True))
        weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y * grad_factor), 3, keepdims=True))

        # [batch, height/scale, width/scale, 1]
        smoothness_x = disp_gradients_x * weights_x
        smoothness_y = disp_gradients_y * weights_y

        smoothness_x = 0.5 * tf.reduce_mean(tf.abs(smoothness_x), axis=[1, 2, 3])
        smoothness_y = 0.5 * tf.reduce_mean(tf.abs(smoothness_y), axis=[1, 2, 3])
        smoothness = smoothness_x + smoothness_y
        return smoothness


class StereoDepthLoss(PhotometricLoss):
    def __init__(self, method, scale_weights):
        super().__init__(method, scale_weights)

    def __call__(self, features, predictions, augm_data):
        """
        desciptions of inputs are available in 'TotalLoss.append_data()'
        :return: photo_loss [batch]
        """
        # synthesize left image from right image
        loss_left = self.stereo_photometric_loss(synth_target_ms=augm_data["stereo_synth_ms"],
                                                 target_ms=augm_data["target_ms"])
        # synthesize right image from left image
        loss_right = self.stereo_photometric_loss(synth_target_ms=augm_data["stereo_synth_ms_R"],
                                                  target_ms=augm_data["target_ms_R"],
                                                  suffix="_R")
        # [2, scales, batch] -> [scales, batch]
        losses = [loss_left, loss_right]
        losses = layers.Lambda(lambda x: tf.reduce_sum(x, axis=0), name="st.photo_LR_sum")(losses)
        # weighted sum over scales: [scales, batch] -> [batch]
        return self.merge_multi_scale_losses(losses, "st.photo_loss_sum")

    def stereo_photometric_loss(self, synth_target_ms, target_ms, suffix=""):
        """
        synthesize image from source to target
        :param synth_target_ms: list of [batch, height/scale, width/scale, 1]
        :param target_ms: list of [batch, height/scale, width/scale, 3]
        :param suffix: "" if right to left, else "_R"
        :return losses [scales, batch]
        """
        losses = []
        for i, (synth_img_sc, target_img_sc) in enumerate(zip(synth_target_ms, target_ms)):
            loss = layers.Lambda(lambda inputs: self.photometric_loss(inputs[0], inputs[1]),
                                 name=f"photo_loss_{i}" + suffix)([synth_img_sc, target_img_sc])
            losses.append(loss)
        return losses


class StereoPoseLoss(LossBase):
    def __call__(self, features, predictions, augm_data):
        pose_lr_pred = predictions["pose_LR"]
        pose_rl_pred = predictions["pose_RL"]
        pose_lr_true_mat = features["stereo_T_LR"]
        pose_lr_true_mat = tf.expand_dims(pose_lr_true_mat, axis=1)
        pose_rl_true_mat = tf.linalg.inv(pose_lr_true_mat)
        pose_lr_true = cp.pose_matr2rvec_batch(pose_lr_true_mat)
        pose_rl_true = cp.pose_matr2rvec_batch(pose_rl_true_mat)
        # loss: [batch, numsrc]
        loss = tf.keras.losses.MSE(pose_lr_true, pose_lr_pred) + tf.keras.losses.MSE(pose_rl_true, pose_rl_pred)
        # loss: [batch]
        loss = tf.reduce_mean(loss, axis=1)
        return loss


class FlowWarpLossMultiScale(PhotometricLoss):
    def __init__(self, method, scale_weights, key_suffix=""):
        super().__init__(method, scale_weights, key_suffix)
        self.scale_wegihts = ()

    def __call__(self, features, predictions, augm_data):
        """
        desciptions of inputs are available in 'TotalLoss.append_data()'
        :return: photo_loss [batch]
        """
        flow_target_ms = augm_data["flow_target_ms" + self.key_suffix]
        # warp a target from 4 sources and 4 flows in 4 level scales
        warped_target_ms = augm_data["warped_target_ms" + self.key_suffix]

        losses = []
        for i, (warp_target, orig_target) in enumerate(zip(warped_target_ms, flow_target_ms)):
            loss = layers.Lambda(lambda inputs: self.photometric_loss(inputs[0], inputs[1]),
                                 name=f"flow_warp_loss_{i}" + self.key_suffix)([warp_target, orig_target])
            losses.append(loss)
        # losses: [loss at each scale] -> sum: scalar
        name = "flow_warp_loss_sum" + self.key_suffix
        # weighted sum over scales: [scales, batch] -> [batch]
        return self.merge_multi_scale_losses(losses, name)


class L2Regularizer(LossBase):
    def __init__(self, weights_to_regularize):
        self.weights = weights_to_regularize

    def __call__(self, features, predictions, augm_data):
        loss = 0
        for weight in self.weights:
            loss += tf.nn.l2_loss(weight)
        # scalar -> [batch]
        batch = features["image5d"].get_shape()[0]
        loss_batch = tf.tile([loss], [batch])
        return loss_batch


# ===== TEST FUNCTIONS

import numpy as np


def test_average_pool_3d():
    print("\n===== start test_average_pool_3d")
    ksize = [1, 3, 3]
    average_pool = tf.keras.layers.AveragePooling3D(pool_size=ksize, strides=1, padding="SAME")
    for i in range(10):
        x = tf.random.normal((8, 4, 100, 100, 3))
        y = tf.random.normal((8, 4, 100, 100, 3))
        mu_x = average_pool(x)
        mu_y = average_pool(y)
        npx = x.numpy()
        npy = y.numpy()
        npmux = mu_x.numpy()
        npmuy = mu_y.numpy()
        print(i, "mean x", npx[0, 0, 10:13, 10:13, 1].mean(), npmux[0, 0, 11, 11, 1],
              "mean y", npy[0, 0, 10:13, 10:13, 1].mean(), npmuy[0, 0, 11, 11, 1])
        assert np.isclose(npx[0, 0, 10:13, 10:13, 1].mean(), npmux[0, 0, 11, 11, 1])
        assert np.isclose(npy[0, 0, 10:13, 10:13, 1].mean(), npmuy[0, 0, 11, 11, 1])

    print("!!! test_average_pool_3d passed")


# in this function, "tf.nn.avg_pool()" works fine with gradient tape in eager mode
def test_gradient_tape():
    print("\n===== start test_gradient_tape")
    out_dim = 10
    x = tf.cast(np.random.uniform(0, 1, (200, 100, 100, 3)), tf.float32)
    y = tf.cast(np.random.uniform(0, 1, (200, out_dim)), tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(100).batch(8)

    input_layer = tf.keras.layers.Input(shape=(100, 100, 3))
    z = tf.keras.layers.Conv2D(64, 3)(input_layer)
    z = tf.keras.layers.Conv2D(out_dim, 3)(z)

    # EXECUTE
    z = tf.nn.avg_pool(z, ksize=[3, 3], strides=[1, 1], padding="SAME")
    z = tf.keras.layers.GlobalAveragePooling2D()(z)
    model = tf.keras.Model(inputs=input_layer, outputs=z)
    optimizer = tf.keras.optimizers.SGD()

    for i, (xi, yi) in enumerate(dataset):
        train_model(model, optimizer, xi, yi)
        uf.print_progress_status(f"optimizing... {i}")


def train_model(model, optimizer, xi, yi):
    with tf.GradientTape() as tape:
        yi_hat = model(xi)
        loss = tf.keras.losses.MeanSquaredError()(yi, yi_hat)

    grad = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grad, model.trainable_weights))


if __name__ == "__main__":
    test_average_pool_3d()
    test_gradient_tape()
