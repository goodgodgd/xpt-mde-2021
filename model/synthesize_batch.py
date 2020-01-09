import os.path as op
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import settings
import utils.convert_pose as cp
from config import opts
from tfrecords.tfrecord_reader import TfrecordGenerator
from utils.decorators import ShapeCheck


@ShapeCheck
def synthesize_batch_multi_scale(src_img_stacked, intrinsic, pred_depth_ms, pred_pose):
    """
    :param src_img_stacked: [batch, height*num_src, width, 3]
    :param intrinsic: [batch, 3, 3]
    :param pred_depth_ms: predicted depth in multi scale, list of [batch, height/scale, width/scale, 1]}
    :param pred_pose: predicted source pose in twist form [batch, num_src, 6]
    :return: reconstructed target view in multi scale, list of [batch, height/scale, width/scale, 3]}
    """
    width_ori = src_img_stacked.get_shape().as_list()[2]
    # convert pose vector to transformation matrix
    poses_matr = layers.Lambda(lambda pose: cp.pose_rvec2matr_batch(pose),
                               name="pose2matrix")(pred_pose)
    recon_images = []
    for depth_sc in pred_depth_ms:
        batch, height_sc, width_sc, _ = depth_sc.get_shape().as_list()
        scale = int(width_ori // width_sc)
        # adjust intrinsic upto scale
        intrinsic_sc = layers.Lambda(lambda intrin: scale_intrinsic(intrin, scale),
                                     name=f"scale_intrin_sc{scale}")(intrinsic)
        # reorganize source images: [batch, 4, height, width, 3]
        source_images_sc = layers.Lambda(lambda image: reshape_source_images(image, scale),
                                         name=f"reorder_source_sc{scale}")(src_img_stacked)
        # print(f"[synthesize_batch_multi_scale] {scale}, source_images_sc {source_images_sc.get_shape()}, "
        #       f"depth_sc {depth_sc.get_shape()}, poses_matr {poses_matr.get_shape()}, ")
        recon_image_sc = synthesize_batch_view(source_images_sc, depth_sc, poses_matr,
                                               intrinsic_sc, suffix=f"sc{scale}")
        recon_images.append(recon_image_sc)

    return recon_images


def scale_intrinsic(intrinsic, scale):
    batch = intrinsic.get_shape().as_list()[0]
    scaled_part = tf.slice(intrinsic, (0, 0, 0), (-1, 2, -1))
    scaled_part = scaled_part / scale
    const_part = tf.tile(tf.constant([[[0, 0, 1]]], dtype=tf.float32), (batch, 1, 1))
    scaled_intrinsic = tf.concat([scaled_part, const_part], axis=1)
    return scaled_intrinsic


@ShapeCheck
def reshape_source_images(src_img_stacked, scale):
    """
    :param src_img_stacked: [batch, height*num_src, width, 3]
    :param scale: scale to reduce image size
    :return: reorganized source images [batch, num_src, height/scale, width/scale, 3]
    """
    num_src = (opts.SNIPPET_LEN - 1)
    # resize image
    scheight, scwidth = (int(opts.IM_HEIGHT // scale), int(opts.IM_WIDTH // scale))
    scaled_image = tf.image.resize(src_img_stacked, size=(scheight*num_src, scwidth), method="bilinear")
    # reorganize scaled images: (4*height/scale,) -> (4, height/scale)
    source_images = tf.reshape(scaled_image, shape=(opts.BATCH_SIZE, num_src, scheight, scwidth, 3))
    return source_images


@ShapeCheck
def synthesize_batch_view(src_image, tgt_depth, pose, intrinsic, suffix):
    """
    src_image, tgt_depth and intrinsic are scaled
    :param src_image: source image nearby the target image [batch, num_src, height, width, 3]
    :param tgt_depth: depth map of the target image in meter scale [batch, height, width, 1]
    :param pose: pose matrices that transform points from target to source frame [batch, num_src, 4, 4]
    :param intrinsic: camera projection matrix [batch, 3, 3]
    :param suffix: suffix to tensor name
    :return: synthesized target image [batch, num_src, height, width, 3]
    """
    _, _, height, width, _ = src_image.get_shape().as_list()
    src_pixel_coords = layers.Lambda(lambda inputs: warp_pixel_coords(inputs, height, width),
                                     name="warp_pixel_"+suffix)\
                                    ([tgt_depth, pose, intrinsic])
    tgt_image_synthesized = layers.Lambda(lambda inputs:
                                          reconstruct_bilinear_interp(inputs[0], inputs[1], inputs[2]),
                                          name="recon_interp_"+suffix)\
                                         ([src_pixel_coords, src_image, tgt_depth])
    return tgt_image_synthesized


def warp_pixel_coords(inputs, height, width):
    tgt_depth, pose, intrinsic = inputs
    tgt_pixel_coords = pixel_meshgrid(height, width)
    tgt_cam_coords = pixel2cam(tgt_pixel_coords, tgt_depth, intrinsic)
    src_cam_coords = transform_to_source(tgt_cam_coords, pose)
    src_pixel_coords = cam2pixel(src_cam_coords, intrinsic)
    return src_pixel_coords


def pixel_meshgrid(height, width, stride=1):
    """
    :return: pixel coordinates like vectors of (u,v,1) [3, height*width]
    """
    v = np.linspace(0, height-stride, int(height//stride)).astype(np.float32)
    u = np.linspace(0, width-stride,  int(width//stride)).astype(np.float32)
    ugrid, vgrid = tf.meshgrid(u, v)
    uv = tf.stack([ugrid, vgrid], axis=0)
    uv = tf.reshape(uv, (2, -1))
    num_pts = uv.get_shape().as_list()[1]
    uv = tf.concat([uv, tf.ones((1, num_pts), tf.float32)], axis=0)
    return uv


def pixel2cam(pixel_coords, depth, intrinsic):
    """
    :param pixel_coords: (u,v,1) [3, height*width]
    :param depth: [batch, height, width, 1]
    :param intrinsic: [batch, 3, 3]
    :return: 3D points like (x,y,z,1) in target frame [batch, 4, height*width]
    """
    batch = depth.get_shape().as_list()[0]
    depth = tf.reshape(depth, (batch, 1, -1))

    # calc sum of products over specified dimension
    # cam_coords[i, j, k] = inv(intrinsic)[i, j, :] dot pixel_coords[:, k]
    # [batch, 3, height*width] = [batch, 3, 3] x [3, height*width]
    cam_coords = tf.tensordot(tf.linalg.inv(intrinsic), pixel_coords, [[2], [0]])

    # [batch, 3, height*width] = [batch, 3, height*width] * [batch, 3, height*width]
    cam_coords *= depth
    # num_pts = height * width
    num_pts = cam_coords.get_shape().as_list()[2]
    # make homogeneous coordinates
    cam_coords = tf.concat([cam_coords, tf.ones((batch, 1, num_pts), tf.float32)], axis=1)
    return cam_coords


@ShapeCheck
def transform_to_source(tgt_coords, t2s_pose):
    """
    :param tgt_coords: target frame coordinates like (x,y,z,1) [batch, 4, height*width]
    :param t2s_pose: pose matrices that transform points from target to source frame [batch, num_src, 4, 4]
    :return: transformed points in source frame like (x,y,z,1) [batch, num_src, 4, height*width]
    """
    num_src = t2s_pose.get_shape().as_list()[1]
    tgt_coords_expand = tf.expand_dims(tgt_coords, 1)
    tgt_coords_expand = tf.tile(tgt_coords_expand, (1, num_src, 1, 1))
    # [batch, num_src, 4, height*width] = [batch, num_src, 4, 4] x [batch, num_src, 4, height*width]
    src_coords = tf.matmul(t2s_pose, tgt_coords_expand)
    return src_coords


def cam2pixel(cam_coords, intrinsic):
    """
    :param cam_coords: 3D points in source frame (x,y,z,1) [batch, num_src, 4, height*width]
    :param intrinsic: intrinsic camera matrix [batch, 3, 3]
    :return: projected pixel coordinates on source image plane (u,v,1) [batch, num_src, 3, height*width]
    """
    batch, num_src, _, length = cam_coords.get_shape().as_list()
    intrinsic_expand = tf.expand_dims(intrinsic, 1)
    # [batch, num_src, 3, 3]
    intrinsic_expand = tf.tile(intrinsic_expand, (1, num_src, 1, 1))

    # [batch, num_src, 3, height*width] = [batch, num_src, 3, 3] x [batch, num_src, 3, height*width]
    point_coords = tf.slice(cam_coords, (0, 0, 0, 0), (-1, -1, 3, -1))
    pixel_coords = tf.matmul(intrinsic_expand, point_coords)
    # pixel_coords = tf.reshape(pixel_coords, (batch, num_src, 3, length))
    # normalize scale
    pixel_scales = pixel_coords[:, :, 2:3, :]
    pixel_coords = pixel_coords / (pixel_scales + 1e-10)
    return pixel_coords


@ShapeCheck
def reconstruct_bilinear_interp(pixel_coords, image, depth):
    """
    :param pixel_coords: floating-point pixel coordinates (u,v,1) [batch, num_src, 3, height*width]
    :param image: source image [batch, num_src, height, width, 3]
    :param depth: target depth image [batch, height, width, 1]
    :return: reconstructed image [batch, num_src, height, width, 3]
    """
    batch, num_src, height, width, _ = image.get_shape().as_list()
    padded_image = zero_pad_image(image)

    # adjust pixel coordinates for padded image
    pixel_coords_pad = shift_and_clip_pixels(pixel_coords, height, width)

    # pixel_floorceil[batch, num_src, :, i] = (u_ceil, u_floor, v_ceil, v_floor)
    pixel_floorceil = neighbor_int_pixels(pixel_coords_pad, height, width)

    # weights[batch, num_src, :, i] = (w_uf_vf, w_uf_vc, w_uc_vf, w_uc_vc)
    weights = calc_neighbor_weights([pixel_coords_pad, pixel_floorceil])

    # sampled_image[batch, num_src, :, height, width, 3] =
    # (im_uf_vf, im_uf_vc, im_uc_vf, im_uc_vc)
    sampled_images = sample_neighbor_images([padded_image, pixel_floorceil])

    # sampled_view = sampled_images.numpy().astype(np.uint8)
    # cv2.imshow("sampled", sampled_view[1, 0, 0].reshape((height, width, 3)))

    # recon_image[batch, num_src, height*width, 3]
    flat_image = merge_images([sampled_images, weights])

    flat_image = erase_invalid_pixels([flat_image, depth])
    recon_image = tf.reshape(flat_image, shape=(batch, num_src, height, width, 3))
    return recon_image


@ShapeCheck
def zero_pad_image(image):
    """
    :param image: [batch, num_src, height, width, 3]
    :return: [batch, num_src, height+2, width+2, 3]
    """
    # pad 1 pixel around image
    top_pad, bottom_pad, left_pad, right_pad = (1, 1, 1, 1)
    paddings = tf.constant([[0, 0], [0, 0], [top_pad, bottom_pad], [left_pad, right_pad], [0, 0]])
    padded_image = tf.pad(image, paddings, "CONSTANT")
    batch, num_src, height, width, _ = image.get_shape().as_list()
    padded_image = tf.reshape(padded_image, (batch, num_src, height+2, width+2, 3))
    # print("zero pad\n", padded_image[0, 0, 0:5, 0:5, 0])
    return padded_image


@ShapeCheck
def shift_and_clip_pixels(pixel_coords, height, width):
    """
    transform pixel coordinates into the 1-pixel padded image
    :param pixel_coords: (u, v, 1) [batch, num_src, 3, height*width]
    :param height: image height
    :param width: image width
    :return: (u, v) [batch, num_src, 2, height*width]
    """
    u = tf.slice(pixel_coords, (0, 0, 0, 0), (-1, -1, 1, -1))
    u = tf.clip_by_value(u + 1, 0, width + 1)
    v = tf.slice(pixel_coords, (0, 0, 1, 0), (-1, -1, 1, -1))
    v = tf.clip_by_value(v + 1, 0, height + 1)
    adjusted_pixels = tf.concat([u, v], axis=2)
    return adjusted_pixels


def neighbor_int_pixels(pixel_coords, height, width):
    """
    :param pixel_coords: (u, v) [batch, num_src, 2, height*width]
    :param height: image height
    :param width: image width
    :return: (u_floor, u_ceil, v_floor, v_ceil) [batch, num_src, 4, height*width]
    """
    u = tf.slice(pixel_coords, (0, 0, 0, 0), (-1, -1, 1, -1))
    u_floor = tf.clip_by_value(tf.floor(u), 0, width+1)
    u_ceil = tf.clip_by_value(u_floor + 1, 0, width+1)
    v = tf.slice(pixel_coords, (0, 0, 1, 0), (-1, -1, 1, -1))
    v_floor = tf.clip_by_value(tf.floor(v), 0, height+1)
    v_ceil = tf.clip_by_value(v_floor + 1, 0, height+1)
    pixel_floorceil = tf.concat([u_floor, u_ceil, v_floor, v_ceil], axis=2)
    return pixel_floorceil


def calc_neighbor_weights(inputs):
    pixel_coords, pixel_floorceil = inputs
    """
    pixel_coords: (u, v) (float) [batch, num_src, 2, height*width]
    pixel_floorceil: (u_floor, u_ceil, v_floor, v_ceil) (int) [batch, num_src, 4, height*width]
    return: 4 neighbor pixel weights (w_uf_vf, w_uf_vc, w_uc_vf, w_uc_vc) 
            [batch, num_src, 4, height*width]
    """
    ui, vi = (0, 1)
    uf, uc, vf, vc = (0, 1, 2, 3)
    w_uf = pixel_floorceil[:, :, uc:uc + 1, :] - pixel_coords[:, :, ui:ui + 1, :]
    w_uc = pixel_coords[:, :, ui:ui + 1, :] - pixel_floorceil[:, :, uf:uf + 1, :]
    w_vf = pixel_floorceil[:, :, vc:vc + 1, :] - pixel_coords[:, :, vi:vi + 1, :]
    w_vc = pixel_coords[:, :, vi:vi + 1, :] - pixel_floorceil[:, :, vf:vf + 1, :]
    w_ufvf = w_uf * w_vf
    w_ufvc = w_uf * w_vc
    w_ucvf = w_uc * w_vf
    w_ucvc = w_uc * w_vc
    weights = tf.concat([w_ufvf, w_ufvc, w_ucvf, w_ucvc], axis=2)
    return weights


@ShapeCheck
def sample_neighbor_images(inputs):
    padded_image, pixel_floorceil = inputs
    """
    padded_image: [batch, num_src, height+2, width+2, 3]
    pixel_floorceil: (u_floor, u_ceil, v_floor, v_ceil) [batch, num_src, 4, height*width]
    return: flattened sampled image [(batch, num_src, 4, height*width, 3)]
    """
    pixel_floorceil = tf.cast(pixel_floorceil, tf.int32)
    uf = tf.squeeze(tf.slice(pixel_floorceil, (0, 0, 0, 0), (-1, -1, 1, -1)), axis=2)
    uc = tf.squeeze(tf.slice(pixel_floorceil, (0, 0, 1, 0), (-1, -1, 1, -1)), axis=2)
    vf = tf.squeeze(tf.slice(pixel_floorceil, (0, 0, 2, 0), (-1, -1, 1, -1)), axis=2)
    vc = tf.squeeze(tf.slice(pixel_floorceil, (0, 0, 3, 0), (-1, -1, 1, -1)), axis=2)

    """
    CAUTION: `tf.gather_nd` looks preferable over `tf.gather`
    however, `tf.gather_nd` raises error that some shape is unknown
    `tf.gather_nd` has no problem with eagerTensor (which has specific values)
    but raises error with Tensor (placeholder from tf.keras.layers.Input())
    It seems to be a bug.
    Suprisingly, `tf.gather` works nicely with 'Tensor'
    """
    # eager_tensor = False
    # if eager_tensor:
    #     # flatten image: [batch, num_src, height_pad*width_path, 3]
    #     batch, num_src, height_pad, width_pad, _ = padded_image.get_shape().as_list()
    #     padded_image_flat = tf.reshape(padded_image, shape=(batch, num_src, height_pad*width_pad, 3))
    #     print(f"padded_image_flat {padded_image_flat.get_shape()}")
    #     ufvf = vf * width_pad + uf
    #     ufvc = vc * width_pad + uf
    #     ucvf = vf * width_pad + uc
    #     ucvc = vc * width_pad + uc
    #
    #     # imflat_ufvf: (batch, num_src, height*width, 3)
    #     imflat_ufvf = tf.gather(padded_image_flat, ufvf, axis=2, batch_dims=2)
    #     imflat_ufvc = tf.gather(padded_image_flat, ufvc, axis=2, batch_dims=2)
    #     imflat_ucvf = tf.gather(padded_image_flat, ucvf, axis=2, batch_dims=2)
    #     imflat_ucvc = tf.gather(padded_image_flat, ucvc, axis=2, batch_dims=2)
    #
    # else:

    # tf.stack([uf, vf]): [batch, num_src, height*width, 2(u,v)]
    imflat_ufvf = tf.gather_nd(padded_image, tf.stack([vf, uf], axis=-1), batch_dims=2)
    imflat_ufvc = tf.gather_nd(padded_image, tf.stack([vc, uf], axis=-1), batch_dims=2)
    imflat_ucvf = tf.gather_nd(padded_image, tf.stack([vf, uc], axis=-1), batch_dims=2)
    imflat_ucvc = tf.gather_nd(padded_image, tf.stack([vc, uc], axis=-1), batch_dims=2)

    # sampled_images: (batch, num_src, 4, height*width, 3)
    sampled_images = tf.stack([imflat_ufvf, imflat_ufvc, imflat_ucvf, imflat_ucvc], axis=2,
                              name="stack_samples")
    return sampled_images


def merge_images(inputs):
    sampled_images, weights = inputs
    """
    sampled_images: flattened sampled image [batch, num_src, 4, height*width, 3]
    weights: 4 neighbor pixel weights (w_uf_vf, w_uf_vc, w_uc_vf, w_uc_vc) 
             [batch, num_src, 4, height*width]
    return: merged_flat_image, [batch, num_src, height*width, 3]
    """
    # expand dimension to channel
    weights = tf.expand_dims(weights, -1)
    weighted_image = sampled_images * weights
    merged_flat_image = tf.reduce_sum(weighted_image, axis=2)
    return merged_flat_image


@ShapeCheck
def erase_invalid_pixels(inputs):
    flat_image, depth = inputs
    """
    flat_image: [batch, num_src, height*width, 3]
    depth: target view depth [batch, height, width, 1]
    return: [batch, num_src, height*width, 3]
    """
    batch, width, height, _ = depth.get_shape().as_list()
    # depth_vec [batch, height*width, 1]
    depth_vec = tf.reshape(depth, shape=(batch, -1, 1))
    depth_vec = tf.expand_dims(depth_vec, 1)
    # depth_vec [batch, 1, height*width, 1]
    depth_invalid_mask = tf.math.equal(depth_vec, 0)
    flat_image = tf.where(depth_invalid_mask, tf.constant(0, dtype=tf.float32), flat_image)
    return flat_image


# --------------------------------------------------------------------------------
# TESTS



def test_synthesize_batch_view():
    tfrgen = TfrecordGenerator(op.join(opts.DATAPATH_TFR, "kitti_raw_test"))
    dataset = tfrgen.get_generator()
    scale = 1
    for x, y in dataset:
        for key, value in x.items():
            print(f"x shape and type: {key}={x[key].shape}, {x[key].dtype}")

        target_view = x['image'][:, opts.IM_HEIGHT*4:opts.IM_HEIGHT*5, :, :]
        target_view = target_view.numpy().astype(np.uint8)

        source_image_sc = reshape_source_images(x['image'], scale)
        print("source shape", source_image_sc.get_shape())
        source_view = source_image_sc.numpy().astype(np.uint8)

        batch, num_src, height, width, _ = source_image_sc.get_shape().as_list()
        depth_sc = tf.image.resize(x['depth_gt'], size=(height, width), method="nearest")
        print("depth shape", x['depth_gt'].get_shape(), depth_sc.get_shape())
        depth_view = depth_sc.numpy()
        cv2.imshow("depth", depth_view[1])

        intrinsic_sc = scale_intrinsic(x['intrinsic'], scale)
        print(f"original intrinsic \n{x['intrinsic'][1]} \nscaled intrinsic \n{intrinsic_sc[1]}")

        pose = x['pose_gt']
        pose_view = pose.numpy()
        print(f"pose shape and data: {pose_view.shape} \n{pose_view[1, 0]}")

        recon_images = synthesize_batch_view(source_image_sc, depth_sc, x['pose_gt'],
                                             intrinsic_sc, "recon")

        print("reconstructed image shape:", recon_images.get_shape(), recon_images.dtype)
        recon_view = recon_images.numpy().astype(np.uint8)

        result = np.concatenate([target_view[1], source_view[1, 0], recon_view[1, 0]], axis=0)
        cv2.imshow("target_source_recon", result)
        cv2.waitKey()


def load_data_batch():
    srcinds = [1430, 1430, 1450, 1450]
    tgtind = 1440
    batch = 8
    src_images = []
    for si in srcinds:
        image = cv2.imread(f"samples/color/{si:05d}.jpg")
        src_images.append(image)
    src_images = np.stack(src_images, axis=0)
    src_images = np.expand_dims(src_images, 0)
    src_images = np.tile(src_images, (batch, 1, 1, 1, 1))
    print("src_images", src_images.shape)
    cv2.imshow("src_image", src_images[0, 0])
    src_images = tf.constant(src_images, tf.float32)

    tgt_image = cv2.imread(f"samples/color/{tgtind:05d}.jpg")
    cv2.imshow("tgt_image", tgt_image)

    tgt_pose = np.loadtxt(f"samples/pose/pose_{tgtind:05d}.txt")
    src_poses = []
    for si in srcinds:
        src_pose = np.loadtxt(f"samples/pose/pose_{si:05d}.txt")
        t2s_pose = np.matmul(np.linalg.inv(src_pose), tgt_pose)
        src_poses.append(t2s_pose)
    src_poses = np.stack(src_poses, axis=0)
    src_poses = np.expand_dims(src_poses, 0)
    src_poses = np.tile(src_poses, (batch, 1, 1, 1))
    print("src_poses", src_poses.shape)
    src_poses = tf.constant(src_poses, tf.float32)

    tgt_depth = cv2.imread(f"samples/depth/{tgtind:05d}.png", cv2.IMREAD_ANYDEPTH)
    tgt_depth = tgt_depth/1000.
    height, width = tgt_depth.shape
    tgt_depth = tgt_depth.reshape((1, height, width, 1))
    tgt_depth = np.tile(tgt_depth, (batch, 1, 1, 1))
    print("target depth", tgt_depth.shape)
    tgt_depth = tf.constant(tgt_depth, tf.float32)

    intrinsic = np.loadtxt("samples/intrinsic.txt")
    intrinsic = np.expand_dims(intrinsic, 0)
    intrinsic = np.tile(intrinsic, (batch, 1, 1))
    intrinsic = tf.constant(intrinsic, tf.float32)

    return src_images, src_poses, tgt_depth, intrinsic


def test_synthesize_batch_view_aug_icl():
    src_images, src_poses, tgt_depth, intrinsic = load_data_batch()
    recon_images = synthesize_batch_view(src_images, tgt_depth, src_poses, intrinsic)
    recon_view = recon_images.numpy()
    recon_img = recon_view[1, 1]
    cv2.imshow("reconstructed", recon_img.astype(np.uint8))
    cv2.waitKey()


def test_gather():
    # params = tf.keras.layers.Input(shape=(4, 5), batch_size=3)
    # params = tf.Variable(initial_value=np.ones((3, 4, 5)), shape=(3, 4, 5))
    params = tf.constant(np.arange(1, 13).reshape((2, 3, 2)))
    print("params", type(params), params.get_shape())
    print(params)
    indices = tf.constant([[0, 1], [1, 2]], dtype=tf.int64)
    values = tf.gather(params, indices, axis=1, batch_dims=1)
    print("values", values.get_shape())
    print(values)


def test():
    np.set_printoptions(precision=3, suppress=True, linewidth=100)
    test_scale_intrinsic()
    test_pixel2cam()
    test_transform_to_source()
    test_pixel_weighting()
    test_gather()
    test_synthesize_batch_view()
    # test_synthesize_batch_view_aug_icl()


if __name__ == "__main__":
    test()
