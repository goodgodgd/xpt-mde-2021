import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa

from config import opts
from utils.decorators import shape_check
from model.synthesize.bilinear_interp import BilinearInterpolation


class FlowWarpMultiScale:
    def __init__(self):
        # shape is scaled from the original shape, height = original_height / scale
        self.batch, self.height, self.width, self.numsrc, self.scale = (0, 0, 0, 0, 0)
        self.suffix = ""

    @shape_check
    def __call__(self, src_img_stacked, flow_ms):
        """
        :param src_img_stacked: source images stacked vertically [batch, height*numsrc, width, 3]
        :param flow_ms: predicted optical flow from source to target in multi scale,
                        list of [batch, numsrc, height/scale, width/scale, 2] (scale: 1, 2, 4, 8)
        :return: reconstructed target view in multi scale, list of [batch, numsrc, height/scale, width/scale, 3]}
        """
        warped_targets = []
        for flow_sc in flow_ms:
            src_img_sc = self.reshape_source_images(src_img_stacked, flow_sc)
            pixel_coords_sc = self.flow_to_pixel_coordinates(flow_sc)
            # construct a new graph for each scale
            warp_target_sc = BilinearInterpolation()(src_img_sc, pixel_coords_sc)
            warped_targets.append(warp_target_sc)
        return warped_targets

    @shape_check
    def reshape_source_images(self, src_img_stacked, flow_sc):
        """
        :param src_img_stacked [batch, height*numsrc, width, 3]
        :param flow_sc [batch, numsrc, height/scale, width/scale, 2]
        :return: reorganized source images [batch, numsrc, height/scale, width/scale, 3]
        """
        batch, numsrc, height, width, _ = flow_sc.get_shape()
        _, stacked_height, width_orig, _ = src_img_stacked.get_shape()
        height_orig = stacked_height // numsrc
        # reshape image -> (batch*numsrc, height_orig, width_orig, 3)
        source_images = tf.reshape(src_img_stacked, shape=(batch * numsrc, height_orig, width_orig, 3))
        # resize image (scaled) -> (batch*numsrc, height, width, 3)
        scaled_image = tf.image.resize(source_images, size=(height, width), method="bilinear")
        # reshape image -> (batch, numsrc, height, width, 3)
        scaled_image = tf.reshape(scaled_image, shape=(batch, numsrc, height, width, 3))
        return scaled_image

    def flow_to_pixel_coordinates(self, flow):
        """
        :param flow: optical flow [batch, numsrc, height/scale, width/scale, 2]
        :return: pixel_coords: source image pixel coordinates [batch, numsrc, 2, height/scale*width/scale]
        """
        batch, numsrc, height, width, _ = flow.get_shape()
        u = tf.range(0, width, 1, dtype=tf.float32)
        v = tf.range(0, height, 1, dtype=tf.float32)
        ugrid, vgrid = tf.meshgrid(u, v)
        uvgrid = tf.stack([ugrid, vgrid], axis=0)
        # uvgrid -> [1, 1, 2, height*width]
        uvgrid = tf.reshape(uvgrid, (1, 1, 2, -1))

        # uvflow -> [batch, numsrc, height*width, 2]
        uvflow = tf.reshape(flow, (batch, numsrc, -1, 2))
        # uvflow -> [batch, numsrc, 2, height*width]
        uvflow = tf.transpose(uvflow, perm=[0, 1, 3, 2])

        # add flow to basic grid
        pixel_coords = uvgrid - uvflow
        return pixel_coords


# ==================================================
import os.path as op
import cv2


def test_reshape_source_images():
    print("===== start test_reshape_source_images")
    opts.ENABLE_SHAPE_DECOR = True
    image_path = op.join(opts.DATAPATH_SRC, "kitti_odom_train", "11", "000002.png")
    src_img = cv2.imread(image_path)
    cv2.imshow("src_img", src_img)
    cv2.waitKey(10)
    bat_img = tf.constant([src_img, src_img, src_img], dtype=tf.float32)
    print("src_img shape", bat_img.shape)

    # take only left image
    left_img = bat_img[:, :, :opts.IM_WIDTH]
    print("left_img shape", left_img.shape)
    cv2.imshow("left_img", left_img[0].numpy().astype(np.uint8))
    cv2.waitKey(10)

    # reshape image
    batch = 3
    numsrc = 5
    height, width = opts.IM_HEIGHT, opts.IM_WIDTH

    # EXECUTE
    rsp_img = tf.reshape(left_img, (batch*numsrc, height, width, 3))

    print("rsp_img shape", rsp_img.shape)
    assert np.isclose(rsp_img[1].numpy(), src_img[height:height*2, :width]).all()
    print("!!! test_reshape_source_images passed")

    cv2.imshow("rsp_img", rsp_img[0].numpy().astype(np.uint8))
    cv2.waitKey(0)


if __name__ == "__main__":
    test_reshape_source_images()
