import tensorflow as tf
from tensorflow.keras import layers
from model.synthesize_batch import synthesize_batch_multi_scale

import settings
from config import opts
import utils.convert_pose as cp
import utils.util_funcs as uf
import evaluate.eval_funcs as ef
from utils.decorators import ShapeCheck


def compute_loss_vode(predictions, features):
    """
    :param predictions: {"disp_ms": .., "pose": ..}
        disp_ms: multi scale disparity, list of [batch, height/scale, width/scale, 1]
        pose: 6-DoF poses [batch, num_src, 6]
    :param features: {"image": .., "pose_gt": .., "depth_gt": .., "intrinsic": ..}
        image: stacked image [batch, height*snippet_len, width, 3]
        intrinsic: camera projection matrix [batch, 3, 3]
    """
    stacked_image = features['image']
    intrinsic = features['intrinsic']
    source_image, target_image = uf.split_into_source_and_target(stacked_image)

    pred_disp_ms = predictions['disp_ms']
    pred_pose = predictions['pose']
    pred_depth_ms = disp_to_depth(pred_disp_ms)

    target_ms = multi_scale_like(target_image, pred_disp_ms)

    synth_target_ms = synthesize_batch_multi_scale(source_image, intrinsic, pred_depth_ms, pred_pose)
    photo_loss = photometric_loss_multi_scale(synth_target_ms, target_ms)
    height_orig = target_image.get_shape().as_list()[2]
    smooth_loss = smootheness_loss_multi_scale(pred_disp_ms, target_ms, height_orig)
    loss = layers.Lambda(lambda losses: tf.add(losses[0], opts.SMOOTH_WEIGHT * losses[1]),
                         name="train_loss")([photo_loss, smooth_loss])
    return loss


def compute_metric_pose(pose_pred, pose_true_mat):
    """
    :param pose_pred: 6-DoF poses [batch, num_src, 6]
    :param pose_true_mat: 4x4 transformation matrix [batch, num_src, 4, 4]
    """
    pose_pred_mat = cp.pose_rvec2matr_batch(pose_pred)
    trj_err = ef.calc_trajectory_error_tensor(pose_pred_mat, pose_true_mat)
    rot_err = ef.calc_rotational_error_tensor(pose_pred_mat, pose_true_mat)
    return tf.reduce_mean(trj_err), tf.reduce_mean(rot_err)


def disp_to_depth(disp_ms):
    target_ms = []
    for i, disp in enumerate(disp_ms):
        target = layers.Lambda(lambda dis: 1./dis, name=f"todepth_{i}")(disp)
        target_ms.append(target)
    return target_ms


def multi_scale_like(image, disp_ms):
    """
    :param image: [batch, height, width, 3]
    :param disp_ms: list of [batch, height/scale, width/scale, 1]
    :return: image_ms: list of [batch, height/scale, width/scale, 3]
    """
    image_ms = []
    for i, disp in enumerate(disp_ms):
        batch, height_sc, width_sc, _ = disp.get_shape().as_list()
        image_sc = layers.Lambda(lambda img: tf.image.resize(img, size=(height_sc, width_sc), method="bilinear"),
                                 name=f"target_resize_{i}")(image)
        image_ms.append(image_sc)
    return image_ms


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
    photo_loss = layers.Lambda(lambda data: tf.reduce_sum(tf.stack(data, axis=0), axis=0),
                               name="photo_loss_sum")(losses)
    return photo_loss


@ShapeCheck
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
        scale = height_orig // image.get_shape().as_list()[1]
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
import os.path as op
import numpy as np
import cv2
from tfrecords.tfrecord_reader import TfrecordGenerator


def test_photometric_loss_quality():
    print("===== start test_photometric_loss_quality")
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
        target_ms = multi_scale_like(target_image, depth_gt_ms)
        batch, height, width, _ = target_image.get_shape().as_list()

        synth_target_ms = synthesize_batch_multi_scale(source_image, intrinsic, depth_gt_ms, pose_gt)

        srcimgs = uf.to_uint8_image(source_image).numpy()[0]
        srcimg0 = srcimgs[0:height]
        srcimg3 = srcimgs[height*3:height*4]

        losses = []
        for scale, synt_target, orig_target in zip([1, 2, 4, 8], synth_target_ms, target_ms):
            # EXECUTE
            loss = photometric_loss(synt_target, orig_target)
            losses.append(loss)

            recon_target = uf.to_uint8_image(synt_target).numpy()
            recon0 = cv2.resize(recon_target[0, 0], (width, height), interpolation=cv2.INTER_NEAREST)
            recon3 = cv2.resize(recon_target[0, 3], (width, height), interpolation=cv2.INTER_NEAREST)
            target = uf.to_uint8_image(orig_target).numpy()[0]
            target = cv2.resize(target, (width, height), interpolation=cv2.INTER_NEAREST)
            view = np.concatenate([target, srcimg0, recon0, srcimg3, recon3], axis=0)
            print(f"1/{scale} scale, photo loss:", loss)
            cv2.imshow("photo loss", view)
            cv2.waitKey()

        losses = tf.stack(losses, axis=0)
        photo_loss = tf.reduce_sum(losses, axis=0)
        print("all photometric loss:", losses)
        print("batch mean photometric loss:", photo_loss)
        print("scale mean photometric loss:", tf.reduce_sum(losses, axis=1))
        if i > 3:
            break

    cv2.destroyAllWindows()
    print("!!! test_photometric_loss_quality passed")


def test_photometric_loss_quantity():
    print("===== start test_photometric_loss_quantity")
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
        target_ms = multi_scale_like(target_image, depth_gt_ms)

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
        cv2.waitKey()
        if i > 3:
            break

    cv2.destroyAllWindows()
    print("!!! test_photometric_loss_quantity passed")


def test_photo_loss(source_image, intrinsic, depth_gt_ms, pose_gt, target_ms):
    synth_target_ms = synthesize_batch_multi_scale(source_image, intrinsic, depth_gt_ms, pose_gt)

    losses = []
    for scale, synt_target, orig_target in zip([1, 2, 4, 8], synth_target_ms, target_ms):
        # EXECUTE
        loss = photometric_loss(synt_target, orig_target)
        losses.append(loss)

        if scale == 1:
            recon_target = uf.to_uint8_image(synt_target).numpy()
            recon_image = cv2.resize(recon_target[0, 0], (opts.IM_WIDTH, opts.IM_HEIGHT), interpolation=cv2.INTER_NEAREST)

    losses = tf.stack(losses, axis=0)
    batch_loss = tf.reduce_sum(losses, axis=0)
    scale_loss = tf.reduce_sum(losses, axis=1)
    print("all photometric loss:", losses)
    print("batch mean photometric loss:", batch_loss)
    print("scale mean photometric loss:", scale_loss)
    return batch_loss, scale_loss, recon_image


def test_smootheness_loss_quantity():
    print("===== start test_smootheness_loss_quantity")
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
        target_ms = multi_scale_like(target_image, depth_gt_ms)

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
        cv2.waitKey()
        if i > 3:
            break

    cv2.destroyAllWindows()
    print("!!! test_smootheness_loss_quantity passed")


def test_depth_to_disp(depth_ms):
    disp_ms = []
    for i, depth in enumerate(depth_ms):
        disp = layers.Lambda(lambda dep: tf.where(dep > 0.1, 1./dep, 0), name=f"todisp_{i}")(depth)
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
    # test_photometric_loss_quality()
    # test_photometric_loss_quantity()
    test_smootheness_loss_quantity()


if __name__ == "__main__":
    test()
