import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from utils.decorators import shape_check
from model.synthesize.bilinear_interp import BilinearInterpolation
from utils.convert_pose import pose_rvec2matr_batch_tf


class SynthesizeMultiScale:
    @shape_check
    def __call__(self, source_image, intrinsic, pred_depth_ms, pred_pose):
        """
        :param source_image: source images stacked vertically [batch, numsrc, height, width, 3]
        :param intrinsic: [batch, 3, 3]
        :param pred_depth_ms: predicted target depth in multi scale, list of [batch, height/scale, width/scale, 1]}
        :param pred_pose: predicted source pose in twist vector for each source frame [batch, numsrc, 6]
                        it transforms target points to source frame
        :return: reconstructed target view in multi scale, list of [batch, numsrc, height/scale, width/scale, 3]}
        """
        # convert pose vector to transformation matrix
        poses_matr = layers.Lambda(lambda pose: pose_rvec2matr_batch_tf(pose),
                                   name="pose2matrix")(pred_pose)
        synth_targets = []
        for depth_sc in pred_depth_ms:
            synth_target_sc = SynthesizeSingleScale()(source_image, intrinsic, depth_sc, poses_matr)
            synth_targets.append(synth_target_sc)

        return synth_targets


class SynthesizeSingleScale:
    def __init__(self, shape=(0, 0, 0), numsrc=0, scale=0):
        # shape is scaled from the original shape, height = original_height / scale
        self.batch, self.height_sc, self.width_sc = shape
        self.numsrc = numsrc
        self.scale = scale

    def __call__(self, source_image, intrinsic, depth_sc, poses_matr):
        """
        :param source_image: stacked source images [batch, numsrc, height, width, 3]
        :param intrinsic: intrinsic parameters [batch, 3, 3]
        :param depth_sc: scaled predicted depth for target image, [batch, height/scale, width/scale, 1]
        :param poses_matr: predicted source pose in matrix form [batch, numsrc, 4, 4]
        :return: reconstructed target view in scale, [batch, numsrc, height/scale, width/scale, 3]
        """
        suffix = f"_sc{self.scale}"
        self.read_shape(source_image, depth_sc)
        # adjust intrinsic upto scale
        intrinsic_sc = layers.Lambda(lambda intrin: self.scale_intrinsic(intrin, self.scale),
                                     name=f"scale_intrin" + suffix)(intrinsic)
        # resize source images: [batch, numsrc, height/scale, width/scale, 3]
        source_images_sc = layers.Lambda(lambda image: self.resize_source_images(image),
                                         name=f"resize_source" + suffix)(source_image)
        # reconstruct target view from source images
        recon_image_sc = self.synthesize_batch_view(source_images_sc, depth_sc, poses_matr,
                                                    intrinsic_sc, suffix=f"sc{self.scale}")
        return recon_image_sc

    @shape_check
    def read_shape(self, source_image, depth_sc):
        batch, self.numsrc, height_orig, _, _ = source_image.get_shape()
        self.batch, self.height_sc, self.width_sc, _ = depth_sc.get_shape()
        self.scale = int(height_orig // self.height_sc)

    def scale_intrinsic(self, intrinsic, scale):
        scaled_part = tf.slice(intrinsic, (0, 0, 0), (-1, 2, -1))
        scaled_part = scaled_part / scale
        const_part = tf.tile(tf.constant([[[0, 0, 1]]], dtype=tf.float32), (self.batch, 1, 1))
        scaled_intrinsic = tf.concat([scaled_part, const_part], axis=1)
        return scaled_intrinsic

    @shape_check
    def resize_source_images(self, source_image):
        """
        :param source_image: [batch, numsrc, height, width, 3]
        :return: reorganized source images [batch, numsrc, height/scale, width/scale, 3]
        """
        batch, numsrc, height_orig, width_orig, _ = source_image.get_shape()
        source_image = tf.reshape(source_image, shape=(batch*numsrc, height_orig, width_orig, 3))
        # resize image (scaled) -> (batch*numsrc, height/scale, width/scale, 3)
        source_image = tf.image.resize(source_image, size=(self.height_sc, self.width_sc), method="bilinear")
        # reorganize scaled images -> (batch, numsrc, height/scale, width/scale, 3)
        source_image = tf.reshape(source_image, shape=(batch, numsrc, self.height_sc, self.width_sc, 3))
        return source_image

    @shape_check
    def synthesize_batch_view(self, src_image, tgt_depth, pose, intrinsic, suffix):
        """
        src_image, tgt_depth and intrinsic are scaled
        :param src_image: source image nearby the target image [batch, numsrc, height, width, 3]
        :param tgt_depth: depth map of the target image in meter scale [batch, height, width, 1]
        :param pose: pose matrices that transform points from target to source frame [batch, numsrc, 4, 4]
        :param intrinsic: camera projection matrix [batch, 3, 3]
        :param suffix: suffix to tensor name
        :return: synthesized target image [batch, numsrc, height, width, 3]
        """
        src_pixel_coords = layers.Lambda(lambda inputs: self.warp_pixel_coords(inputs, self.height_sc, self.width_sc),
                                         name="warp_pixel_" + suffix)([tgt_depth, pose, intrinsic])
        tgt_image_synthesized = layers.Lambda(lambda inputs:
                                              BilinearInterpolation()(inputs[0], inputs[1], inputs[2]),
                                              name="recon_interp_" + suffix)(
                                              [src_image, src_pixel_coords, tgt_depth])
        return tgt_image_synthesized

    def warp_pixel_coords(self, inputs, height, width):
        tgt_depth, pose, intrinsic = inputs
        tgt_pixel_coords = self.pixel_meshgrid(height, width)
        tgt_cam_coords = self.pixel2cam(tgt_pixel_coords, tgt_depth, intrinsic)
        src_cam_coords = self.transform_to_source(tgt_cam_coords, pose)
        src_pixel_coords = self.cam2pixel(src_cam_coords, intrinsic)
        return src_pixel_coords

    def pixel_meshgrid(self, height, width, stride=1):
        """
        :return: pixel coordinates like vectors of (u,v,1) [3, height*width]
        """
        v = np.linspace(0, height - stride, int(height // stride)).astype(np.float32)
        u = np.linspace(0, width - stride, int(width // stride)).astype(np.float32)
        ugrid, vgrid = tf.meshgrid(u, v)
        uv = tf.stack([ugrid, vgrid], axis=0)
        uv = tf.reshape(uv, (2, -1))
        uv = tf.concat([uv, tf.ones((1, height*width), tf.float32)], axis=0)
        return uv

    def pixel2cam(self, pixel_coords, depth, intrinsic):
        """
        :param pixel_coords: (u,v,1) [3, height*width]
        :param depth: [batch, height, width, 1]
        :param intrinsic: [batch, 3, 3]
        :return: 3D points like (x,y,z,1) in target frame [batch, 4, height*width]
        """
        depth = tf.reshape(depth, (self.batch, 1, -1))

        # calc sum of products over specified dimension
        # cam_coords[i, j, k] = inv(intrinsic)[i, j, :] dot pixel_coords[:, k]
        # [batch, 3, height*width] = [batch, 3, 3] x [3, height*width]
        cam_coords = tf.tensordot(tf.linalg.inv(intrinsic), pixel_coords, [[2], [0]])

        # [batch, 3, height*width] = [batch, 3, height*width] * [batch, 1, height*width]
        cam_coords *= depth
        # num_pts = height * width
        num_pts = cam_coords.get_shape().as_list()[2]
        # make homogeneous coordinates
        cam_coords = tf.concat([cam_coords, tf.ones((self.batch, 1, num_pts), tf.float32)], axis=1)
        return cam_coords

    @shape_check
    def transform_to_source(self, tgt_coords, t2s_pose):
        """
        :param tgt_coords: target frame coordinates like (x,y,z,1) [batch, 4, height*width]
        :param t2s_pose: pose matrices that transform points from target to source frame [batch, numsrc, 4, 4]
        :return: transformed points in source frame like (x,y,z,1) [batch, numsrc, 4, height*width]
        """
        tgt_coords_expand = tf.expand_dims(tgt_coords, 1)
        tgt_coords_expand = tf.tile(tgt_coords_expand, (1, self.numsrc, 1, 1))
        # [batch, numsrc, 4, height*width] = [batch, numsrc, 4, 4] x [batch, numsrc, 4, height*width]
        src_coords = tf.matmul(t2s_pose, tgt_coords_expand)
        return src_coords

    def cam2pixel(self, cam_coords, intrinsic):
        """
        :param cam_coords: 3D points in source frame (x,y,z,1) [batch, numsrc, 4, height*width]
        :param intrinsic: intrinsic camera matrix [batch, 3, 3]
        :return: projected pixel coordinates on source image plane (u,v,1) [batch, numsrc, 3, height*width]
        """
        intrinsic_expand = tf.expand_dims(intrinsic, 1)
        # [batch, numsrc, 3, 3]
        intrinsic_expand = tf.tile(intrinsic_expand, (1, self.numsrc, 1, 1))

        # [batch, numsrc, 3, height*width] = [batch, numsrc, 3, 3] x [batch, numsrc, 3, height*width]
        point_coords = tf.slice(cam_coords, (0, 0, 0, 0), (-1, -1, 3, -1))
        pixel_coords = tf.matmul(intrinsic_expand, point_coords)
        # pixel_coords = tf.reshape(pixel_coords, (batch, numsrc, 3, length))
        # normalize scale
        pixel_scales = pixel_coords[:, :, 2:3, :]
        pixel_coords = pixel_coords / (pixel_scales + 1e-10)
        return pixel_coords
