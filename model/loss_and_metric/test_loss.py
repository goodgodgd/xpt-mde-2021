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
from model.synthesize.synthesize_base import SynthesizeMultiScale
import model.loss_and_metric.losses as ls

WAIT_KEY = 0


def test_photometric_loss_quality(suffix=""):
    """
    gt depth와 gt pose를 입력했을 때 스케일 별로 복원되는 이미지를 정성적으로 확인하고
    복원된 이미지로부터 계산되는 photometric loss를 확인
    assert 없음
    """
    print("\n===== start test_photometric_loss_quality")
    dataset = TfrecordGenerator(op.join(opts.DATAPATH_TFR, "kitti_raw_test")).get_generator()

    for i, features in enumerate(dataset):
        print("\n--- fetch a batch data")
        stacked_image = features["image" + suffix]
        intrinsic = features["intrinsic" + suffix]
        depth_gt = features["depth_gt" + suffix]
        pose_gt = features["pose_gt" + suffix]

        target_ms = []
        for i, disp in enumerate(disp_ms):
            target = layers.Lambda(lambda dis: 1. / dis, name=f"todepth_{i}")(disp)
            target_ms.append(target)

        # identity pose results in NaN data
        pose_gt_np = pose_gt.numpy()
        for pose_seq in pose_gt_np:
            for pose in pose_seq:
                assert not np.isclose(np.identity(4, dtype=np.float), pose).all()

        source_image, target_image = uf.split_into_source_and_target(stacked_image)
        depth_gt_ms = uf.multi_scale_depths(depth_gt, [1, 2, 4, 8])
        pose_gt = cp.pose_matr2rvec_batch(pose_gt)
        target_ms = uf.multi_scale_like(target_image, depth_gt_ms)
        batch, height, width, _ = target_image.get_shape().as_list()

        synth_target_ms = SynthesizeMultiScale()(source_image, intrinsic, depth_gt_ms, pose_gt)

        srcimgs = uf.to_uint8_image(source_image).numpy()[0]
        srcimg0 = srcimgs[0:height]
        srcimg3 = srcimgs[height*3:height*4]

        losses = []
        for scale, synt_target, orig_target in zip([1, 2, 4, 8], synth_target_ms, target_ms):
            # EXECUTE
            loss = ls.photometric_loss_l1(synt_target, orig_target)
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


def test_photometric_loss_quantity(suffix=""):
    """
    gt depth와 gt pose를 입력했을 때 나오는 photometric loss와
    gt pose에 노이즈를 추가하여 나오는 photometric loss를 비교
    두 가지 pose로 복원된 영상을 눈으로 확인하고 gt 데이터의 loss가 더 낮음을 확인 (assert)
    """
    print("\n===== start test_photometric_loss_quantity")
    dataset = TfrecordGenerator(op.join(opts.DATAPATH_TFR, "kitti_raw_test")).get_generator()

    for i, features in enumerate(dataset):
        print("\n--- fetch a batch data")
        stacked_image = features["image" + suffix]
        intrinsic = features["intrinsic" + suffix]
        depth_gt = features["depth_gt" + suffix]
        pose_gt = features["pose_gt" + suffix]
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
    synth_target_ms = SynthesizeMultiScale()(source_image, intrinsic, depth_gt_ms, pose_gt)

    losses = []
    recon_image = 0
    for scale, synt_target, orig_target in zip([1, 2, 4, 8], synth_target_ms, target_ms):
        # EXECUTE
        loss = ls.photometric_loss_l1(synt_target, orig_target)
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

    """
    gather additional data required to compute losses
    :param features: {image, intrinsic}
            image: stacked image snippet [batch, snippet_len*height, width, 3]
            intrinsic: camera projection matrix [batch, 3, 3]
    :param predictions: {disp_ms, pose}
            disp_ms: multi scale disparities, list of [batch, height/scale, width/scale, 1]
            pose: poses that transform points from target to source [batch, num_src, 6]
    :return augm_data: {depth_ms, source, target, target_ms, synth_target_ms}
            depth_ms: multi scale depth, list of [batch, height/scale, width/scale, 1]
            source: source frames [batch, num_src*height, width, 3]
            target: target frame [batch, height, width, 3]
            target_ms: multi scale target frame, list of [batch, height/scale, width/scale, 3]
            synth_target_ms: multi scale synthesized target frames generated from each source image,
                            list of [batch, num_src, height/scale, width/scale, 3]
    """

    for i, features in enumerate(dataset):
        print("\n--- fetch a batch data")

        stacked_image = features["image"]
        depth_gt = features["depth_gt"]
        # interpolate depth
        depth_gt = tf.image.resize(depth_gt, size=(int(opts.IM_HEIGHT/2), int(opts.IM_WIDTH/2)), method="bilinear")
        depth_gt = tf.image.resize(depth_gt, size=(opts.IM_HEIGHT, opts.IM_WIDTH), method="bilinear")
        # make multi-scale data
        source_image, target_image = uf.split_into_source_and_target(stacked_image)
        depth_gt_ms = uf.multi_scale_depths(depth_gt, [1, 2, 4, 8])
        disp_gt_ms = tu_depth_to_disp(depth_gt_ms)
        target_ms = uf.multi_scale_like(target_image, depth_gt_ms)

        features = {"image": stacked_image}
        predictions = {"disp_ms": disp_gt_ms}
        augm_data = {"target_ms": target_ms}

        # EXECUTE
        batch_loss_right = tu_smootheness_loss(features, predictions, augm_data)
        print("> batch photometric losses:", batch_loss_right)

        print("> corrupt depth to increase gradient of depth")
        depth_gt = depth_gt.numpy()
        depth_gt_right = np.copy(depth_gt)
        depth_gt_wrong = np.copy(depth_gt)
        depth_gt_wrong[:, 10:200:20] = 0
        depth_gt_wrong[:, 11:200:20] = 0
        depth_gt_wrong[:, 12:200:20] = 0
        depth_gt = tf.constant(depth_gt_wrong, dtype=tf.float32)
        depth_gt_ms = uf.multi_scale_depths(depth_gt, [1, 2, 4, 8])
        disp_gt_ms = tu_depth_to_disp(depth_gt_ms)

        # change prediction
        predictions = {"disp_ms": disp_gt_ms}

        # EXECUTE
        batch_loss_wrong = tu_smootheness_loss(features, predictions, augm_data)

        # TEST
        print("> loss diff: wrong - right =", batch_loss_wrong - batch_loss_right)
        assert np.sum(batch_loss_right.numpy() <= batch_loss_wrong.numpy()).all()

        view = np.concatenate([depth_gt_right[0], depth_gt_wrong[0]], axis=0)
        cv2.imshow("target image corruption", view)
        cv2.waitKey(WAIT_KEY)
        if i > 3:
            break

    cv2.destroyAllWindows()
    print("!!! test_smootheness_loss_quantity passed")


def tu_depth_to_disp(depth_ms):
    disp_ms = []
    for i, depth in enumerate(depth_ms):
        disp = layers.Lambda(lambda dep: tf.where(dep > 0.001, 1./dep, 0), name=f"todisp_{i}")(depth)
        disp_ms.append(disp)
    return disp_ms


@tf.function
def tu_smootheness_loss(features, predictions, augm_data):
    batch_loss_wrong = ls.SmoothenessLossMultiScale()(features, predictions, augm_data)
    return batch_loss_wrong


def test_stereo_loss():
    print("\n===== start test_photometric_loss_quality")
    dataset = TfrecordGenerator(op.join(opts.DATAPATH_TFR, "kitti_raw_test")).get_generator()

    for i, features in enumerate(dataset):
        print("\n--- fetch a batch data")
        stereo_loss = ls.StereoDepthLoss("L1")
        total_loss = ls.TotalLoss()
        predictions = tu_make_prediction(features)
        pred_right = tu_make_prediction(features, "_R")
        predictions.update(pred_right)
        augm_data = total_loss.augment_data(features, predictions)
        augm_data_rig = total_loss.augment_data(features, predictions, "_R")
        augm_data.update(augm_data_rig)

        loss_left, synth_left_ms = \
            stereo_loss.stereo_synthesize_loss(features, predictions, augm_data, augm_data["target_R"], True)
        loss_right, synth_right_ms = \
            stereo_loss.stereo_synthesize_loss(features, predictions, augm_data, augm_data["target"], True, "_R")
        losses = loss_left + loss_right
        batch_loss = layers.Lambda(lambda data: tf.reduce_sum(tf.stack(data, axis=2), axis=[1, 2]),
                                   name="photo_loss_sum")(losses)

        tu_show_synthesize_result(synth_left_ms, augm_data["target"], augm_data["target_R"], "left")
        tu_show_synthesize_result(synth_right_ms, augm_data["target_R"], augm_data["target"], "right")
        print("stereo loss:", batch_loss)
        cv2.waitKey(0)


def tu_make_prediction(features, suffix=""):
    depth = features["depth_gt" + suffix]
    depth_ms = uf.multi_scale_depths(depth, [1, 2, 4, 8])
    disp_ms = []
    for k, depth in enumerate(depth_ms):
        disp = tf.where(depth < 0.00001, 0, 1. / depth)
        disp_ms.append(disp)

    poses = features["pose_gt" + suffix]
    poses = cp.pose_matr2rvec_batch(poses)
    predictions = {"pose" + suffix: poses, "disp_ms" + suffix: disp_ms}
    return predictions


def tu_show_synthesize_result(synth_target_ms, target, source, suffix):
    target_stacked = [target[0].numpy(), source[0].numpy()]
    dstsize = (opts.IM_WIDTH, opts.IM_HEIGHT)
    for target in synth_target_ms:
        target_img = uf.to_uint8_image(target)[0][0].numpy()
        target_rsz = cv2.resize(target_img, dstsize, interpolation=cv2.INTER_NEAREST)
        target_stacked.append(target_rsz)
    target_stacked = np.concatenate(target_stacked, axis=0)
    cv2.imshow("synthesized_" + suffix, target_stacked)


def test():
    # test_photometric_loss_quality("_R")
    # test_photometric_loss_quantity("_R")
    # test_smootheness_loss_quantity()
    test_stereo_loss()


if __name__ == "__main__":
    test()
