import os.path as op
import numpy as np
import tensorflow as tf
import cv2

from config import opts
import model.synthesize_batch as sb
from tfrecords.tfrecord_reader import TfrecordGenerator
import utils.convert_pose as cp
import utils.util_funcs as uf


def test_synthesize_batch_multi_scale():
    dataset = TfrecordGenerator(op.join(opts.DATAPATH_TFR, "kitti_raw_test")).get_generator()

    for features in dataset:
        stacked_image = features['image']
        intrinsic = features['intrinsic']
        depth_gt = features['depth_gt']
        pose_gt = features['pose_gt']
        source_image, target_image = uf.split_into_source_and_target(stacked_image)
        pred_depth_ms = multi_scale_depths(depth_gt, [1, 2, 4, 8])
        pred_pose = cp.pose_matr2rvec_batch(pose_gt)

        synth_target_ms = sb.synthesize_batch_multi_scale(source_image, intrinsic, pred_depth_ms, pred_pose)

        stacked_img = tf.image.convert_image_dtype((stacked_image + 1.) / 2., dtype=tf.uint8).numpy()
        recon_img0 = tf.image.convert_image_dtype((synth_target_ms[0] + 1.) / 2., dtype=tf.uint8).numpy()
        view = np.concatenate([stacked_img[0], recon_img0[0, 3]], axis=0)
        cv2.imshow("reconstructed", view)
        cv2.waitKey()

        # TODO: loss, metric 계산해서 작게 나오는지 확인


def multi_scale_depths(depth, scales):
    """ shape checked!
    :param depth: [batch, height, width, 1]
    :param scales: list of scales
    :return: list of depths [batch, height/scale, width/scale, 1]
    """
    batch, height, width, _ = depth.get_shape().as_list()
    depth_ms = []
    for sc in scales:
        scaled_size = (int(height // sc), int(width // sc))
        scdepth = tf.image.resize(depth, size=scaled_size, method="bilinear")
        depth_ms.append(scdepth)
        print("[multi_scale_depths] scaled depth shape:", scdepth.get_shape().as_list())
    return depth_ms


def test_all():
    np.set_printoptions(precision=4, suppress=True)
    test_synthesize_batch_multi_scale()


if __name__ == "__main__":
    test_all()
