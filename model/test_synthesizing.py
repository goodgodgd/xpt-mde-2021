from tensorflow.keras import layers

from config import opts
from model.synthesize_batch import *
from tfrecords.tfrecord_reader import TfrecordGenerator
import utils.convert_pose as cp
import utils.util_funcs as uf


# TODO: loss, metric 계산해서 작게 나오는지 확인
#   sample_neighbor_images, merge_images 이런건 어떻게 테스트하지?
#   synthesize_batch_view 테스트

def test_synthesize_batch_multi_scale():
    print("===== start test_synthesize_batch_multi_scale")
    dataset = TfrecordGenerator(op.join(opts.DATAPATH_TFR, "kitti_raw_test")).get_generator()

    for i, features in enumerate(dataset):
        print("----- test_synthesize_batch_multi_scale")
        stacked_image = features['image']
        intrinsic = features['intrinsic']
        depth_gt = features['depth_gt']
        pose_gt = features['pose_gt']
        source_image, target_image = uf.split_into_source_and_target(stacked_image)
        pred_depth_ms = multi_scale_depths(depth_gt, [1, 2, 4, 8])
        pred_pose = cp.pose_matr2rvec_batch(pose_gt)

        # EXECUTE
        synth_target_ms = synthesize_batch_multi_scale(source_image, intrinsic, pred_depth_ms, pred_pose)

        # compare target image and reconstructed images
        # recon_img0[0, 0]: reconstructed from the first image
        target_image = uf.to_uint8_image(target_image).numpy()
        source_image = uf.to_uint8_image(source_image).numpy()
        recon_img0 = uf.to_uint8_image(synth_target_ms[0]).numpy()
        recon_img1 = uf.to_uint8_image(synth_target_ms[1]).numpy()
        view = np.concatenate([source_image[0, 0:opts.IM_HEIGHT], target_image[0], recon_img0[0, 0]], axis=0)
        print("Check if all the images are the same")
        cv2.imshow("source, target, and reconstructed", view)
        cv2.imshow("scaled reconstruction", recon_img1[0, 0])
        cv2.waitKey()
        if i >= 3:
            break

    cv2.destroyAllWindows()
    print("!!! test_synthesize_batch_multi_scale passed")


def test_synthesize_batch_view():
    print("===== start test_synthesize_batch_view")
    dataset = TfrecordGenerator(op.join(opts.DATAPATH_TFR, "kitti_raw_test")).get_generator()
    scale_idx = 1

    for i, features in enumerate(dataset):
        stacked_image = features['image']
        intrinsic = features['intrinsic']
        depth_gt = features['depth_gt']
        pose_gt = features['pose_gt']
        source_image, target_image = uf.split_into_source_and_target(stacked_image)
        pred_depth_ms = multi_scale_depths(depth_gt, [1, 2, 4, 8])

        # check only 1 scale
        depth_scaled = pred_depth_ms[scale_idx]
        width_ori = source_image.get_shape().as_list()[2]
        batch, height_sc, width_sc, _ = depth_scaled.get_shape().as_list()
        scale = int(width_ori // width_sc)
        # adjust intrinsic upto scale
        intrinsic_sc = layers.Lambda(lambda intrin: scale_intrinsic(intrin, scale),
                                     name=f"scale_intrin_sc{scale}")(intrinsic)
        # reorganize source images: [batch, 4, height, width, 3]
        srcimg_scaled = layers.Lambda(lambda image: reshape_source_images(image, scale),
                                      name=f"reorder_source_sc{scale}")(source_image)

        recon_image_sc = synthesize_batch_view(srcimg_scaled, depth_scaled, pose_gt,
                                               intrinsic_sc, suffix=f"sc{scale}")

        print("reconstructed image", recon_image_sc.get_shape())
        # convert single target image in batch
        target_image = tf.image.resize(target_image[0], size=recon_image_sc.get_shape().as_list()[2:4], method="bilinear")
        target_image = uf.to_uint8_image(target_image).numpy()
        recon_image_sc = uf.to_uint8_image(recon_image_sc[0]).numpy()
        recon_image_sc = recon_image_sc.reshape((4*height_sc, width_sc, 3))
        view = np.concatenate([target_image, recon_image_sc], axis=0)
        cv2.imshow("synthesize_batch", view)
        cv2.waitKey()
        if i >= 3:
            break

    cv2.destroyAllWindows()


def test_reshape_source_images():
    print("===== start test_reshape_source_images")
    dataset = TfrecordGenerator(op.join(opts.DATAPATH_TFR, "kitti_raw_test")).get_generator()
    dataset = iter(dataset)
    features = next(dataset)
    stacked_image = features['image']
    source_image, target_image = uf.split_into_source_and_target(stacked_image)
    print("batch source image shape", source_image.shape)

    # EXECUTE
    reshaped_image = reshape_source_images(source_image, 2)

    print("reorganized source image shape", reshaped_image.get_shape().as_list())
    reshaped_image = uf.to_uint8_image(reshaped_image).numpy()
    imgidx = 2
    scsize = (int(opts.IM_HEIGHT/2), int(opts.IM_WIDTH/2))
    scaled_image = tf.image.resize(source_image, size=(scsize[0]*4, scsize[1]), method="bilinear")
    scaled_image = uf.to_uint8_image(scaled_image).numpy()
    scaled_image = scaled_image[0, scsize[0]*imgidx:scsize[0]*(imgidx+1)]
    # compare second image in the stacked images
    assert np.isclose(scaled_image, reshaped_image[0, imgidx]).all()

    view = np.concatenate([scaled_image, reshaped_image[0, 1]], axis=0)
    cv2.imshow("original and reshaped", view)
    cv2.waitKey()
    print("!!! test_reshape_source_images passed")
    cv2.destroyAllWindows()


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


def test_scale_intrinsic():
    print("===== start test_scale_intrinsic")
    intrinsic = np.array([8, 0, 4, 0, 8, 4, 0, 0, 1], dtype=np.float32).reshape((1, 3, 3))
    intrinsic = tf.constant(np.tile(intrinsic, (8, 1, 1)))
    scale = 2

    # EXECUTE
    intrinsic_sc = scale_intrinsic(intrinsic, scale)

    print("original intrinsic:", intrinsic[0])
    print("scaled intrinsic:", intrinsic_sc[0])
    assert np.isclose((intrinsic[:, :2, :]/2), intrinsic_sc[:, :2, :]).all()
    assert np.isclose((intrinsic[:, -1, :]), intrinsic_sc[:, -1, :]).all()
    print("!!! test_scale_intrinsic passed")


def test_pixel2cam():
    print("===== start test_pixel2cam")
    batch, height, width = (8, 4, 4)
    tgt_pixel_coords = pixel_meshgrid(height, width)
    tgt_pixel_coords = tf.cast(tgt_pixel_coords, dtype=tf.float32)
    intrinsic = np.array([4, 0, height/2, 0, 4, width/2, 0, 0, 1], dtype=np.float32).reshape((1, 3, 3))
    intrinsic = tf.constant(np.tile(intrinsic, (batch, 1, 1)), dtype=tf.float32)
    depth = tf.ones((batch, height, width), dtype=tf.float32) * 2

    # EXECUTE
    tgt_cam_coords = pixel2cam(tgt_pixel_coords, depth, intrinsic)

    print(tgt_cam_coords[0])
    assert (tgt_cam_coords.get_shape() == (batch, 4, height*width))
    print("!!! test_pixel2cam passed")


def test_transform_to_source():
    print("===== start test_transform_to_source")
    batch, num_pts, num_src = (8, 6, 3)
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
    poses = np.tile(poses, (batch, num_src, 1, 1))
    print(f"poses: {poses.shape}\n{poses[2, 1]}")
    poses = tf.constant(poses, dtype=tf.float32)

    # EXECUTE
    src_coords = transform_to_source(coords, poses)

    print(f"src coordinates: {src_coords.get_shape()}\n{src_coords[2, 1]}")
    assert np.isclose(coords[2, :3]*2 + 1, src_coords[2, 1, :3]).all()
    print("!!! test_transform_to_source passed")


def test_pixel_weighting():
    print("===== start test_pixel_weighting")
    batch, num_src, height, width = (8, 4, 5, 5)
    # create random coordinates
    pixel_coords = np.random.uniform(0.1, 3.9, (batch, num_src, 3, height*width))
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

    # EXECUTE -> pixel_floorceil[batch, num_src, :, i] = [u_ceil, u_floor, v_ceil, v_floor]
    pixel_floorceil = neighbor_int_pixels(pixel_coords, height, width)

    print(f"pixel coords floorceil \n{pixel_floorceil[1, 1, :, :6]}")
    print(np.floor(pixel_coords[1, 1, :2, :6] + 1))
    assert np.isclose(np.floor(pixel_coords[:, :, 0, 2:]), pixel_floorceil[:, :, 0, 2:]).all()
    assert np.isclose(np.ceil(pixel_coords[:, :, 1, 2:]), pixel_floorceil[:, :, 3, 2:]).all()
    print("!!! test neighbor_int_pixels passed")
    print("----- start test calc_neighbor_weights")

    # EXECUTE
    valid_mask = make_valid_mask(pixel_floorceil)

    # EXECUTE -> weights[batch, num_src, :, i] = (w_uf_vf, w_uf_vc, w_uc_vf, w_uc_vc)
    weights = calc_neighbor_weights([pixel_coords, pixel_floorceil, valid_mask])

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
    batch, num_src, height, width = (8, 4, 5, 5)

    pixel_coords = np.meshgrid(np.arange(0, height), np.arange(0, width))
    pixel_coords = np.stack(pixel_coords, axis=0).reshape((1, 1, 2, 5, 5)).astype(np.float32)
    # add x coords by 0.3
    u_add = 1.3
    pixel_coords[0, 0, 0] += u_add
    pixel_coords = np.tile(pixel_coords, (batch, num_src, 1, 1, 1))
    pixel_coords = np.reshape(pixel_coords, (batch, num_src, 2, height*width))
    pixel_coords = tf.constant(pixel_coords)

    # EXECUTE
    pixel_floorceil = neighbor_int_pixels(pixel_coords, height, width)
    # EXECUTE
    mask = make_valid_mask(pixel_floorceil)

    print("valid mask\n", mask[0, 0, 0].numpy().reshape((height, width)))
    expected_mask = np.zeros((batch, num_src, height, width), dtype=np.float)
    expected_mask[:, :, :4, :3] = 1
    expected_mask = expected_mask.reshape((batch, num_src, 1, height*width))
    assert np.isclose(expected_mask, mask).all()
    print("!!! test neighbor_int_pixels and make_valid_mask passed")

    print("----- start test reconstruct_bilinear_interp")
    image = np.meshgrid(np.arange(0, height), np.arange(0, width))[0].reshape((1, 1, height, width, 1))
    image = np.tile(image, (batch, num_src, 1, 1, 3)).astype(np.float32)
    image = tf.constant(image)

    depth = np.ones((height, width)).reshape((1, height, width, 1))
    depth = np.tile(depth, (batch, 1, 1, 1)).astype(np.float32)
    depth = tf.constant(depth)

    # EXECUTE
    recon_image = reconstruct_bilinear_interp(pixel_coords, image, depth)

    expected_image = (image + u_add) * expected_mask.reshape((batch, num_src, height, width, 1))
    print("input image", image[0, 0, :, :, 0])
    print(f"expected image shifted by {u_add} along u-axis", expected_image[0, 0, :, :, 0])
    print("reconstructed image", recon_image[0, 0, :, :, 0])
    assert np.isclose(recon_image, expected_image).all()
    print("!!! test reconstruct_bilinear_interp passed")


def test_all():
    np.set_printoptions(precision=4, suppress=True, linewidth=100)
    # test_synthesize_batch_multi_scale()
    test_synthesize_batch_view()
    test_reshape_source_images()
    # test_scale_intrinsic()
    # test_pixel2cam()
    # test_transform_to_source()
    # test_pixel_weighting()
    # test_reconstruct_bilinear_interp()


if __name__ == "__main__":
    test_all()
