import os.path as op
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import cv2

from config import opts
from model.synthesize.synthesize_base import SynthesizeSingleScale, SynthesizeMultiScale
from model.synthesize.bilinear_interp import BilinearInterpolation
from tfrecords.tfrecord_reader import TfrecordReader
from model.model_util.augmentation import augmentation_factory
import utils.convert_pose as cp
import utils.util_funcs as uf

WAIT_KEY = 0


def test_synthesize_batch_multi_scale():
    """
    gt depth와 gt pose를 입력했을 때 스케일 별로 복원되는 이미지를 정성적으로 확인
    실제 target image와 복원된 "multi" scale target image를 눈으로 비교
    """
    print("===== start test_synthesize_batch_multi_scale")
    dataname, split = "waymo", "train"
    dataset = TfrecordReader(op.join(opts.DATAPATH_TFR, f"{dataname}_{split}")).get_dataset()
    augmenter = augmentation_factory(opts.AUGMENT_PROBS)

    for i, features in enumerate(dataset):
        print("----- test_synthesize_batch_multi_scale")
        features = augmenter(features)
        image5d = features['image5d']
        intrinsic = features['intrinsic']
        depth_gt = features['depth_gt']
        pose_gt = features['pose_gt']
        source_image, target_image = image5d[:, :4], image5d[:, 4]
        depth_gt_ms = uf.multi_scale_depths(depth_gt, [1, 2, 4, 8])
        pred_pose = cp.pose_matr2rvec_batch(pose_gt)
        cv2.imshow("depth", depth_gt[0].numpy())

        # EXECUTE
        synth_target_ms = SynthesizeMultiScale()(source_image, intrinsic, depth_gt_ms, pred_pose)

        # compare target image and reconstructed images
        # recon_img0[0, 0]: reconstructed from the first image
        target_image = uf.to_uint8_image(target_image).numpy()[0]
        source_image = uf.to_uint8_image(source_image).numpy()[0, 0]
        recon_img0 = uf.to_uint8_image(synth_target_ms[0]).numpy()[0, 0]
        recon_img1 = uf.to_uint8_image(synth_target_ms[2]).numpy()[0, 0]
        print("recon image size:", recon_img0.shape, recon_img1.shape, opts.get_img_shape("WH", dataname))
        recon_img1 = cv2.resize(recon_img1, opts.get_img_shape("WH", dataname), cv2.INTER_NEAREST)
        view = np.concatenate([source_image, target_image, recon_img0, recon_img1], axis=0)
        print("Check if all the images are the same")
        cv2.imshow("source, target, and reconstructed", view)
        cv2.waitKey(WAIT_KEY)
        if i >= 3:
            break

    cv2.destroyAllWindows()
    print("!!! test_synthesize_batch_multi_scale passed")


def test_synthesize_batch_view():
    """
    gt depth와 gt pose를 입력했을 때 스케일 별로 복원되는 이미지를 정성적으로 확인
    실제 target image와 복원된 "single" scale target image를 눈으로 비교
    """
    dataset = TfrecordReader(op.join(opts.DATAPATH_TFR, "kitti_raw_test")).get_dataset()

    print("\n===== start test_synthesize_batch_view")
    scale_idx = 1

    for i, features in enumerate(dataset):
        stacked_image = features['image']
        intrinsic = features['intrinsic']
        depth_gt = features['depth_gt']
        pose_gt = features['pose_gt']
        source_image, target_image = uf.split_into_source_and_target(stacked_image)
        depth_gt_ms = uf.multi_scale_depths(depth_gt, [1, 2, 4, 8])
        batch, height, width, _ = target_image.get_shape().as_list()

        # check only 1 scale
        depth_scaled = depth_gt_ms[scale_idx]
        width_ori = source_image.get_shape().as_list()[2]
        batch, height_sc, width_sc, _ = depth_scaled.get_shape().as_list()
        scale = int(width_ori // width_sc)
        # create synthesizer
        synthesizer = SynthesizeSingleScale(shape=(batch, height_sc, width_sc), numsrc=4, scale=scale)
        # adjust intrinsic upto scale
        intrinsic_sc = layers.Lambda(lambda intrin: synthesizer.scale_intrinsic(intrin, scale),
                                     name=f"scale_intrin_sc{scale}")(intrinsic)
        # reorganize source images: [batch, 4, height, width, 3]
        srcimg_scaled = layers.Lambda(lambda image: synthesizer.reshape_source_images(image),
                                      name=f"reorder_source_sc{scale}")(source_image)

        # EXECUTE
        recon_image_sc = synthesizer.synthesize_batch_view(
            srcimg_scaled, depth_scaled, pose_gt, intrinsic_sc, suffix=f"sc{scale}")

        print("reconstructed image shape:", recon_image_sc.get_shape())
        # convert single target image in batch
        target_image = uf.to_uint8_image(target_image[0]).numpy()
        recon_image = uf.to_uint8_image(recon_image_sc[0]).numpy()
        recon_image = recon_image.reshape((4*height_sc, width_sc, 3))
        recon_image = cv2.resize(recon_image, (width, height*4), interpolation=cv2.INTER_NEAREST)
        view = np.concatenate([target_image, recon_image], axis=0)
        cv2.imshow("synthesize_batch", view)
        cv2.waitKey(WAIT_KEY)
        if i >= 3:
            break

    cv2.destroyAllWindows()


def test_reshape_source_images():
    """
    위 아래로 쌓인 원본 이미지를 batch 아래 한 차원을 더 만들어서 reshape이 잘 됐는지 확인(assert)
    """
    print("===== start test_reshape_source_images")
    dataset = TfrecordReader(op.join(opts.DATAPATH_TFR, "kitti_raw_test")).get_dataset()
    dataset = iter(dataset)
    features = next(dataset)
    stacked_image = features['image']
    source_image, target_image = uf.split_into_source_and_target(stacked_image)
    print("batch source image shape", source_image.shape)
    # create synthesizer
    batch, height, width, _ = target_image.get_shape().as_list()
    synthesizer = SynthesizeSingleScale((batch, int(height/2), int(width/2)), 4, 2)

    # EXECUTE
    reshaped_image = synthesizer.reshape_source_images(source_image)

    print("reorganized source image shape", reshaped_image.get_shape().as_list())
    reshaped_image = uf.to_uint8_image(reshaped_image).numpy()
    imgidx = 2
    scsize = opts.get_img_shape("HW", scale_div=2)
    scaled_image = tf.image.resize(source_image, size=(scsize[0]*4, scsize[1]), method="bilinear")
    scaled_image = uf.to_uint8_image(scaled_image).numpy()
    scaled_image = scaled_image[0, scsize[0]*imgidx:scsize[0]*(imgidx+1)]
    # compare second image in the stacked images
    assert np.isclose(scaled_image, reshaped_image[0, imgidx]).all()

    view = np.concatenate([scaled_image, reshaped_image[0, 1]], axis=0)
    cv2.imshow("original and reshaped", view)
    cv2.waitKey(WAIT_KEY)
    print("!!! test_reshape_source_images passed")
    cv2.destroyAllWindows()


def test_scale_intrinsic():
    print("===== start test_scale_intrinsic")
    batch = 8
    intrinsic = np.array([8, 0, 4, 0, 8, 4, 0, 0, 1], dtype=np.float32).reshape((1, 3, 3))
    intrinsic = tf.constant(np.tile(intrinsic, (batch, 1, 1)))
    scale = 2

    # EXECUTE
    intrinsic_sc = SynthesizeSingleScale(shape=(batch, 1, 1)).scale_intrinsic(intrinsic, scale)

    print("original intrinsic:", intrinsic[0])
    print("scaled intrinsic:", intrinsic_sc[0])
    assert np.isclose((intrinsic[:, :2, :]/2), intrinsic_sc[:, :2, :]).all()
    assert np.isclose((intrinsic[:, -1, :]), intrinsic_sc[:, -1, :]).all()
    print("!!! test_scale_intrinsic passed")


def test_pixel2cam():
    print("===== start test_pixel2cam")
    shape = (8, 4, 4)
    batch, height, width = shape
    # create synthesizer
    synthesizer = SynthesizeSingleScale(shape, 4, 1)
    tgt_pixel_coords = synthesizer.pixel_meshgrid(height, width)
    tgt_pixel_coords = tf.cast(tgt_pixel_coords, dtype=tf.float32)
    intrinsic = np.array([4, 0, height/2, 0, 4, width/2, 0, 0, 1], dtype=np.float32).reshape((1, 3, 3))
    intrinsic = tf.constant(np.tile(intrinsic, (batch, 1, 1)), dtype=tf.float32)
    depth = tf.ones((batch, height, width), dtype=tf.float32) * 2

    # EXECUTE
    tgt_cam_coords = synthesizer.pixel2cam(tgt_pixel_coords, depth, intrinsic)

    print(tgt_cam_coords[0])
    assert (tgt_cam_coords.get_shape() == (batch, 4, height*width))
    print("!!! test_pixel2cam passed")


def test_transform_to_source():
    print("===== start test_transform_to_source")
    batch, num_pts, numsrc = (8, 6, 3)
    # create integer coordinates
    coords = np.arange(1, 4*num_pts+1).reshape((num_pts, 4)).T
    coords[3, :] = 1
    coords = np.tile(coords, (batch, 1, 1))
    print(f"coordinates: {coords.shape}\n{coords[2]}")
    coords = tf.constant(coords, dtype=tf.float32)
    # 3 poses to make src_coords = 2*coords + 1
    poses = np.identity(4)*2
    poses[:3, 3] = 1
    poses[3, 3] = 1
    poses = np.tile(poses, (batch, numsrc, 1, 1))
    print(f"poses: {poses.shape}\n{poses[2, 1]}")
    poses = tf.constant(poses, dtype=tf.float32)

    # EXECUTE
    src_coords = SynthesizeSingleScale().transform_to_source(coords, poses)

    print(f"src coordinates: {src_coords.get_shape()}\n{src_coords[2, 1]}")
    assert np.isclose(coords[2, :3]*2 + 1, src_coords[2, 1, :3]).all()
    print("!!! test_transform_to_source passed")


def test_pixel_weighting():
    print("===== start test_pixel_weighting")
    batch, numsrc, height, width = (8, 4, 5, 5)
    # create random coordinates
    pixel_coords = np.random.uniform(0.1, 3.9, (batch, numsrc, 3, height*width))
    # to check 'out of image' pixels
    pixel_coords[:, :, :, 0] = -1.5
    pixel_coords[:, :, :, 1] = 7
    # to check weights
    chk_u, chk_v = 0.2, 0.7
    pixel_coords[:, :, 0, 3] = 2 + chk_u
    pixel_coords[:, :, 1, 3] = 3 + chk_v
    # set z=1 in (u,v,z)
    pixel_coords[:, :, 2, :] = 1
    pixel_coords = tf.constant(pixel_coords, dtype=tf.float32)
    print(f"pixel coords shape: {pixel_coords.get_shape()}")
    print(f"pixel coords original \n{pixel_coords[1, 1, :, :6]}")
    print("----- start test neighbor_int_pixels")

    # EXECUTE -> pixel_floorceil[batch, numsrc, :, i] = [u_ceil, u_floor, v_ceil, v_floor]
    pixel_floorceil = BilinearInterpolation().neighbor_int_pixels(pixel_coords, height, width)

    print(f"pixel coords floorceil \n{pixel_floorceil[1, 1, :, :6]}")
    print(np.floor(pixel_coords[1, 1, :2, :6] + 1))
    assert np.isclose(np.floor(pixel_coords[:, :, 0, 2:]), pixel_floorceil[:, :, 0, 2:]).all()
    assert np.isclose(np.ceil(pixel_coords[:, :, 1, 2:]), pixel_floorceil[:, :, 3, 2:]).all()
    print("!!! test neighbor_int_pixels passed")
    print("----- start test calc_neighbor_weights")

    # EXECUTE
    valid_mask = BilinearInterpolation().make_valid_mask(pixel_floorceil)

    # EXECUTE -> weights[batch, numsrc, :, i] = (w_uf_vf, w_uf_vc, w_uc_vf, w_uc_vc)
    weights = BilinearInterpolation().calc_neighbor_weights([pixel_coords, pixel_floorceil, valid_mask])

    print(f"pixel weights \n{weights[1, 1, :, :6]}")
    assert np.isclose(weights[:, :, 0, 3], (1 - chk_u) * (1 - chk_v)).all()  # ufvf
    assert np.isclose(weights[:, :, 1, 3], (1 - chk_u) * chk_v).all()        # ufvc
    assert np.isclose(weights[:, :, 2, 3], chk_u * (1 - chk_v)).all()        # ucvf
    assert np.isclose(weights[:, :, 3, 3], chk_u * chk_v).all()              # ucvc
    weight_sum = tf.reduce_sum(weights, axis=2)
    weight_sum = weight_sum.numpy()
    print(f"weight sum \n{weight_sum[1, 1, :6]}")
    assert (np.isclose(weight_sum, 0) | np.isclose(weight_sum, 1)).all()
    print("!!! test calc_neighbor_weights passed")


def test_reconstruct_bilinear_interp():
    print("===== start test_reconstruct_bilinear_interp")
    print("----- start test neighbor_int_pixels and make_valid_mask")
    batch, numsrc, height, width = (8, 4, 5, 5)

    pixel_coords = np.meshgrid(np.arange(0, height), np.arange(0, width))
    pixel_coords = np.stack(pixel_coords, axis=0).reshape((1, 1, 2, 5, 5)).astype(np.float32)
    # add x coords by 0.3
    u_add = 1.3
    pixel_coords[0, 0, 0] += u_add
    pixel_coords = np.tile(pixel_coords, (batch, numsrc, 1, 1, 1))
    pixel_coords = np.reshape(pixel_coords, (batch, numsrc, 2, height*width))
    pixel_coords = tf.constant(pixel_coords)

    # EXECUTE
    pixel_floorceil = BilinearInterpolation().neighbor_int_pixels(pixel_coords, height, width)
    # EXECUTE
    mask = BilinearInterpolation().make_valid_mask(pixel_floorceil)

    print("valid mask\n", mask[0, 0, 0].numpy().reshape((height, width)))
    expected_mask = np.zeros((batch, numsrc, height, width), dtype=np.float)
    expected_mask[:, :, :4, :3] = 1
    expected_mask = expected_mask.reshape((batch, numsrc, 1, height*width))
    assert np.isclose(expected_mask, mask).all()
    print("!!! test neighbor_int_pixels and make_valid_mask passed")

    print("----- start test reconstruct_bilinear_interp")
    image = np.meshgrid(np.arange(0, height), np.arange(0, width))[0].reshape((1, 1, height, width, 1))
    image = np.tile(image, (batch, numsrc, 1, 1, 3)).astype(np.float32)
    image = tf.constant(image)

    depth = np.ones((height, width)).reshape((1, height, width, 1))
    depth = np.tile(depth, (batch, 1, 1, 1)).astype(np.float32)
    depth = tf.constant(depth)

    # EXECUTE
    recon_image = BilinearInterpolation()(image, pixel_coords, depth)

    expected_image = (image + u_add) * expected_mask.reshape((batch, numsrc, height, width, 1))
    print("input image", image[0, 0, :, :, 0])
    print(f"expected image shifted by {u_add} along u-axis", expected_image[0, 0, :, :, 0])
    print("reconstructed image", recon_image[0, 0, :, :, 0])
    assert np.isclose(recon_image, expected_image).all()
    print("!!! test reconstruct_bilinear_interp passed")


def test_all():
    np.set_printoptions(precision=4, suppress=True, linewidth=100)
    test_synthesize_batch_multi_scale()
    # test_synthesize_batch_view()
    # test_reshape_source_images()
    # test_scale_intrinsic()
    # test_pixel2cam()
    # test_transform_to_source()
    # test_pixel_weighting()
    # test_reconstruct_bilinear_interp()


if __name__ == "__main__":
    test_all()
