import os.path as op
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers

import settings
from config import opts
import utils.util_funcs as uf
import utils.convert_pose as cp
from tfrecords.tfrecord_reader import TfrecordGenerator
from model.synthesize.synthesize_factory import synthesizer_factory
from model.loss.losses import photometric_loss_l1, smootheness_loss

WAIT_KEY = 200


def test_photometric_loss_quality():
    """
    gt depth와 gt pose를 입력했을 때 스케일 별로 복원되는 이미지를 정성적으로 확인하고
    복원된 이미지로부터 계산되는 photometric loss를 확인
    assert 없음
    """
    print("\n===== start test_photometric_loss_quality")
    dataset = TfrecordGenerator(op.join(opts.DATAPATH_TFR, "kitti_raw_test")).get_generator()

    for i, features in enumerate(dataset):
        print("\n--- fetch a batch data")
        stacked_image = features['image']
        intrinsic = features['intrinsic']
        depth_gt = features['depth_gt']
        pose_gt = features['pose_gt']
        source_image, target_image = uf.split_into_source_and_target(stacked_image)
        depth_gt_ms = uf.multi_scale_depths(depth_gt, [1, 2, 4, 8])
        pose_gt = cp.pose_matr2rvec_batch(pose_gt)
        target_ms = uf.multi_scale_like(target_image, depth_gt_ms)
        batch, height, width, _ = target_image.get_shape().as_list()

        synth_target_ms = synthesizer_factory()(source_image, intrinsic, depth_gt_ms, pose_gt)

        srcimgs = uf.to_uint8_image(source_image).numpy()[0]
        srcimg0 = srcimgs[0:height]
        srcimg3 = srcimgs[height*3:height*4]

        losses = []
        for scale, synt_target, orig_target in zip([1, 2, 4, 8], synth_target_ms, target_ms):
            # EXECUTE
            loss = photometric_loss_l1(synt_target, orig_target)
            losses.append(loss)

            recon_target = uf.to_uint8_image(synt_target).numpy()
            recon0 = cv2.resize(recon_target[0, 0], (width, height), interpolation=cv2.INTER_NEAREST)
            recon3 = cv2.resize(recon_target[0, 3], (width, height), interpolation=cv2.INTER_NEAREST)
            target = uf.to_uint8_image(orig_target).numpy()[0]
            target = cv2.resize(target, (width, height), interpolation=cv2.INTER_NEAREST)
            view = np.concatenate([target, srcimg0, recon0, srcimg3, recon3], axis=0)
            print(f"1/{scale} scale, photo loss:", tf.reduce_sum(loss, axis=1))
            cv2.imshow("photo loss", view)
            cv2.waitKey(WAIT_KEY)

        losses = tf.stack(losses, axis=2)       # [batch, num_src, num_scales]
        print("all photometric loss:", tf.reduce_sum(losses, axis=1))
        print("batch mean photometric loss:", tf.reduce_sum(losses, axis=[1, 2]))
        print("scale mean photometric loss:", tf.reduce_sum(losses, axis=[0, 1]))
        if i > 3:
            break

    cv2.destroyAllWindows()
    print("!!! test_photometric_loss_quality passed")


def test_photometric_loss_quantity():
    """
    gt depth와 gt pose를 입력했을 때 나오는 photometric loss와
    gt pose에 노이즈를 추가하여 나오는 photometric loss를 비교
    두 가지 pose로 복원된 영상을 눈으로 확인하고 gt 데이터의 loss가 더 낮음을 확인 (assert)
    """
    print("\n===== start test_photometric_loss_quantity")
    dataset = TfrecordGenerator(op.join(opts.DATAPATH_TFR, "kitti_raw_test")).get_generator()

    for i, features in enumerate(dataset):
        print("\n--- fetch a batch data")
        stacked_image = features['image']
        intrinsic = features['intrinsic']
        depth_gt = features['depth_gt']
        pose_gt = features['pose_gt']
        source_image, target_image = uf.split_into_source_and_target(stacked_image)
        depth_gt_ms = uf.multi_scale_depths(depth_gt, [1, 2, 4, 8])
        pose_gt = cp.pose_matr2rvec_batch(pose_gt)
        target_ms = uf.multi_scale_like(target_image, depth_gt_ms)

        # EXECUTE
        batch_loss_right, scale_loss_right, recon_image_right = \
            test_photo_loss(source_image, intrinsic, depth_gt_ms, pose_gt, target_ms)

        print("\ncorrupt poses")
        pose_gt = pose_gt.numpy()
        pose_gt = pose_gt + np.random.uniform(-0.2, 0.2, pose_gt.shape)
        pose_gt = tf.constant(pose_gt, dtype=tf.float32)

        # EXECUTE
        batch_loss_wrong, scale_loss_wrong, recon_image_wrong = \
            test_photo_loss(source_image, intrinsic, depth_gt_ms, pose_gt, target_ms)

        # TEST
        print("loss diff: wrong - right =", batch_loss_wrong - batch_loss_right)
        # Due to randomness, allow minority of frames to fail to the test
        assert (np.sum(batch_loss_right.numpy() < batch_loss_wrong.numpy()) > opts.BATCH_SIZE//4)
        assert (np.sum(scale_loss_right.numpy() < scale_loss_wrong.numpy()) > opts.BATCH_SIZE//4)

        target = uf.to_uint8_image(target_image).numpy()[0]
        view = np.concatenate([target, recon_image_right, recon_image_wrong], axis=0)
        cv2.imshow("pose corruption", view)
        cv2.waitKey(WAIT_KEY)
        if i > 3:
            break

    cv2.destroyAllWindows()
    print("!!! test_photometric_loss_quantity passed")


def test_photo_loss(source_image, intrinsic, depth_gt_ms, pose_gt, target_ms):
    synth_target_ms = synthesizer_factory(opts.SYNTHESIZER)(source_image, intrinsic, depth_gt_ms, pose_gt)

    losses = []
    recon_image = 0
    for scale, synt_target, orig_target in zip([1, 2, 4, 8], synth_target_ms, target_ms):
        # EXECUTE
        loss = photometric_loss_l1(synt_target, orig_target)
        losses.append(loss)
        if scale == 1:
            recon_target = uf.to_uint8_image(synt_target).numpy()
            recon_image = cv2.resize(recon_target[0, 0], (opts.IM_WIDTH, opts.IM_HEIGHT), interpolation=cv2.INTER_NEAREST)

    losses = tf.stack(losses, axis=2)   # [batch, num_src, num_scales]
    batch_loss = tf.reduce_sum(losses, axis=[1, 2])
    scale_loss = tf.reduce_sum(losses, axis=[0, 1])
    print("all photometric loss:", losses)
    print("batch mean photometric loss:", batch_loss)
    print("scale mean photometric loss:", scale_loss)
    return batch_loss, scale_loss, recon_image


def test_smootheness_loss_quantity():
    """
    gt depth로부터 계산되는 smootheness loss 비교
    gt depth에 일부를 0으로 처리하여 전체적인 gradient를 높인 depth의 smootheness loss 비교
    두 가지 depth를 눈으로 확인하고 gt 데이터의 loss가 더 낮음을 확인 (assert)
    """
    print("\n===== start test_smootheness_loss_quantity")
    dataset = TfrecordGenerator(op.join(opts.DATAPATH_TFR, "kitti_raw_test")).get_generator()

    for i, features in enumerate(dataset):
        print("\n--- fetch a batch data")
        stacked_image = features['image']
        depth_gt = features['depth_gt']
        # interpolate depth
        depth_gt = tf.image.resize(depth_gt, size=(int(opts.IM_HEIGHT/2), int(opts.IM_WIDTH/2)), method="bilinear")
        depth_gt = tf.image.resize(depth_gt, size=(opts.IM_HEIGHT, opts.IM_WIDTH), method="bilinear")
        # make multi-scale data
        source_image, target_image = uf.split_into_source_and_target(stacked_image)
        depth_gt_ms = uf.multi_scale_depths(depth_gt, [1, 2, 4, 8])
        disp_gt_ms = test_depth_to_disp(depth_gt_ms)
        target_ms = uf.multi_scale_like(target_image, depth_gt_ms)

        # EXECUTE
        batch_loss_right, scale_loss_right = test_smooth_loss(disp_gt_ms, target_ms)

        print("\ncorrupt depth to increase gradient of depth")
        depth_gt = depth_gt.numpy()
        depth_gt_right = np.copy(depth_gt)
        depth_gt_wrong = np.copy(depth_gt)
        depth_gt_wrong[:, 10:200:20] = 0
        depth_gt_wrong[:, 11:200:20] = 0
        depth_gt_wrong[:, 12:200:20] = 0
        depth_gt = tf.constant(depth_gt_wrong, dtype=tf.float32)
        depth_gt_ms = uf.multi_scale_depths(depth_gt, [1, 2, 4, 8])
        disp_gt_ms = test_depth_to_disp(depth_gt_ms)

        # EXECUTE
        batch_loss_wrong, scale_loss_wrong = test_smooth_loss(disp_gt_ms, target_ms)

        # TEST
        print("loss diff: wrong - right =", batch_loss_wrong - batch_loss_right)
        assert np.sum(batch_loss_right.numpy() <= batch_loss_wrong.numpy()).all()
        assert np.sum(scale_loss_right.numpy() <= scale_loss_wrong.numpy()).all()

        view = np.concatenate([depth_gt_right[0], depth_gt_wrong[0]], axis=0)
        cv2.imshow("target image corruption", view)
        cv2.waitKey(WAIT_KEY)
        if i > 3:
            break

    cv2.destroyAllWindows()
    print("!!! test_smootheness_loss_quantity passed")


def test_depth_to_disp(depth_ms):
    disp_ms = []
    for i, depth in enumerate(depth_ms):
        disp = layers.Lambda(lambda dep: tf.where(dep > 0.001, 1./dep, 0), name=f"todisp_{i}")(depth)
        disp_ms.append(disp)
    return disp_ms


def test_smooth_loss(disp_ms, target_ms):
    """
    :param disp_ms: list of [batch, height/scale, width/scale, 1]
    :param target_ms: list of [batch, height/scale, width/scale, 3]
    :return:
    """
    losses = []
    for scale, disp, image in zip([1, 2, 4, 8], disp_ms, target_ms):

        # EXECUTE
        loss = smootheness_loss(disp, image)

        losses.append(loss)

    losses = tf.stack(losses, axis=0)
    batch_loss = tf.reduce_sum(losses, axis=0)
    scale_loss = tf.reduce_sum(losses, axis=1)
    print("all photometric loss:", losses)
    print("batch mean photometric loss:", batch_loss)
    print("scale mean photometric loss:", scale_loss)
    return batch_loss, scale_loss


def test():
    test_photometric_loss_quality()
    test_photometric_loss_quantity()
    test_smootheness_loss_quantity()


if __name__ == "__main__":
    test()
