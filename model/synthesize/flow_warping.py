import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa

from config import opts
from utils.decorators import shape_check


class FlowWarpMultiScale:
    @shape_check
    def __call__(self, src_img_stacked, flow_ms):
        """
        :param src_img_stacked: source images stacked vertically [batch, height*num_src, width, 3]
        :param flow_ms: predicted optical flow from source to target in multi scale,
                        list of [batch, num_src, height/scale, width/scale, 1] (scale: 1, 2, 4, 8)
        :return: reconstructed target view in multi scale, list of [batch, num_src, height/scale, width/scale, 3]}
        """
        warped_targets = []
        for flow_sc in flow_ms:
            # construct a new graph for each scale
            warp_target_sc = FlowWarpSingleScale()(src_img_stacked, flow_sc)
            warped_targets.append(warp_target_sc)
        return warped_targets


class FlowWarpSingleScale:
    def __init__(self):
        # shape is scaled from the original shape, height = original_height / scale
        self.batch, self.height, self.width, self.num_src, self.scale = (0, 0, 0, 0, 0)
        self.suffix = ""

    def __call__(self, src_img_stacked, flow_sc):
        """
        :param src_img_stacked: stacked source images [batch, height*num_src, width, 3]
        :param flow_sc: scaled predicted optical flow for target image, [batch, num_src, height/scale, width/scale, 1]
        :return: warped target view in scale, [batch, num_src, height/scale, width/scale, 3]
        """
        self.read_shape(src_img_stacked, flow_sc)
        # resize and reshape source images -> [batch*num_src, height/scale, width/scale, 3]
        src_img_scaled = layers.Lambda(lambda image: self.reshape_source_images(image),
                                       name=f"flow_reorder_src" + self.suffix)(src_img_stacked)
        warped_image = self.warp_image(src_img_scaled, flow_sc)
        return warped_image

    @shape_check
    def read_shape(self, src_img_stacked, flow_sc):
        batch_size, stacked_height, width_orig, _ = src_img_stacked.get_shape()
        self.batch, self.num_src, self.height, self.width, _ = flow_sc.get_shape()
        self.scale = int(width_orig / self.width)
        self.suffix = f"_sc{self.scale}"

    @shape_check
    def reshape_source_images(self, src_img_stacked):
        """
        :param src_img_stacked: [batch, height*num_src, width, 3]
        :return: reorganized source images [batch, num_src, height/scale, width/scale, 3]
        """
        batch_size, stacked_height, width_orig, _ = src_img_stacked.get_shape()
        height_orig = stacked_height // self.num_src
        # reshape image -> (batch*num_src, height_orig, width_orig, 3)
        source_images = tf.reshape(src_img_stacked, shape=(self.batch * self.num_src, height_orig, width_orig, 3))
        # resize image (scaled) -> (batch*num_src, height, width, 3)
        scaled_image = tf.image.resize(source_images, size=(self.height, self.width), method="bilinear")
        return scaled_image

    def warp_image(self, src_img, flow_sc):
        """
        :param src_img: reorganized source images [batch*num_src, height/scale, width/scale, 3]
        :param flow_sc: optical flow [batch, num_src, height/scale, width/sclae, 2]
        :return:
        """
        # reshape flow -> [batch*num_src, height/scale, width/scale, 2]
        flow_reshaped = tf.reshape(flow_sc, (self.batch * self.num_src, self.height, self.width, 2))
        # warped target view from source images -> [batch*num_src, height/scale, width/scale, 3]
        warped_image = tfa.image.dense_image_warp(src_img, flow_reshaped, name=f"warped" + self.suffix)
        # reshape warped image-> [batch, num_src, height/scale, width/scale, 3]
        warped_image = tf.reshape(warped_image, (self.batch, self.num_src, self.height, self.width, 3))
        return warped_image


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
    num_src = 5
    height, width = opts.IM_HEIGHT, opts.IM_WIDTH

    # EXECUTE
    rsp_img = tf.reshape(left_img, (batch*num_src, height, width, 3))

    print("rsp_img shape", rsp_img.shape)
    assert np.isclose(rsp_img[1].numpy(), src_img[height:height*2, :width]).all()
    print("!!! test_reshape_source_images passed")

    cv2.imshow("rsp_img", rsp_img[0].numpy().astype(np.uint8))
    cv2.waitKey(0)


if __name__ == "__main__":
    test_reshape_source_images()
