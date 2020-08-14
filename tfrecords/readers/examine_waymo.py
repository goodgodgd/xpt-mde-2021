import os
import tensorflow as tf
import math
import numpy as np
from scipy import sparse
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

from model.synthesize.synthesize_base import SynthesizeSingleScale
import utils.util_funcs as uf


def set_configs():
    np.set_printoptions(precision=3, suppress=True)
    # set gpu configs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def get_dataset():
    np.set_printoptions(precision=3, suppress=True, linewidth=100)
    file_pattern = f"/media/ian/IanBook/datasets/waymo/training_0005/*.tfrecord"
    filenames = tf.io.gfile.glob(file_pattern)
    print("[tfrecord reader]", file_pattern, filenames)
    dataset = tf.data.TFRecordDataset(filenames, compression_type='')
    return dataset


IM_ID = 0


def show_front_image_depth_pose():
    dataset = get_dataset()
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        front_image = tf.image.decode_jpeg(frame.images[IM_ID].image)
        # rgb to bgr
        front_image = front_image.numpy()[:, :, [2, 1, 0]]
        print("image[0]", frame.images[IM_ID].name, front_image.shape, front_image.dtype)
        print("image[0] pose", tf.reshape(frame.images[IM_ID].pose.transform, (4, 4)))

        depth_map = get_depth_map(frame)

        dstshape = (front_image.shape[0] // 2, front_image.shape[1] // 2)
        view = make_view("image", front_image, depth_map, dstshape)
        cv2.imshow("image", view)
        key = cv2.waitKey()
        if key == ord("q"):
            break


def get_depth_map(frame):
    (range_images, camera_projections, range_image_top_pose) = \
        frame_utils.parse_range_image_and_camera_projection(frame)
    """
    points[i]: LiDAR i의 xyz 좌표들 [N, 3] 
    cp_points[i]: LiDAR i를 camera에 projection한 이미지 좌표들 [camidx1, iy1, ix1, camidx2, iy2, ix2]
    """
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose)

    height, width = (frame.context.camera_calibrations[0].height, frame.context.camera_calibrations[0].width)

    # xyz points in vehicle frame
    points_veh = np.concatenate(points, axis=0)
    # cp_points: (Nx6) [cam_id, ix, iy, cam_id, ix, iy]
    cp_points = np.concatenate(cp_points, axis=0)[:, :3]
    print("points all:", points_veh.shape, "cp_points", cp_points.shape, np.max(cp_points, axis=0))

    # extract LiDAR points projected to camera[IM_ID]
    print("camera name:", frame.images[IM_ID].name)
    mask = np.equal(cp_points[:, 0], frame.images[IM_ID].name)
    cp_points = cp_points[mask]
    points_veh = points_veh[mask]
    print("cam1 points all:", points_veh.shape, "cam1 cp_points", cp_points.shape)

    # transform points from vehicle to camera1
    intrin = frame.context.camera_calibrations[0].intrinsic
    cam1_K = np.array([ [intrin[0], 0, intrin[2]], [0, intrin[1], intrin[3]], [0, 0, 1] ])
    cam1_T_C2V = tf.reshape(frame.context.camera_calibrations[0].extrinsic.transform, (4, 4)).numpy()
    cam1_T_V2C = np.linalg.inv(cam1_T_C2V)
    print("intrinsic:\n", intrin)
    print("camera mat:\n", cam1_K)
    print("extrinsic:\n", cam1_T_V2C)
    points_veh_homo = np.concatenate((points_veh, np.ones((points_veh.shape[0], 1))), axis=1)
    points_veh_homo = points_veh_homo.T
    print("points_veh_homo minmax\n", points_veh_homo[:, 100:-1:2000])
    points_cam_homo = cam1_T_V2C @ points_veh_homo
    print("points_cam_homo minmax\n", points_cam_homo[:, 100:-1:2000])
    points_depth = points_cam_homo[0]

    # project points into image
    # normalize depth to 1
    points_cam = points_cam_homo[:3]
    points_cam_norm = points_cam / points_cam[0:1]
    print("points_cam_norm\n", np.min(points_cam_norm, axis=1), np.max(points_cam_norm, axis=1))
    # 3D Y axis = left = -image x,  ix = -Y*fx + cx
    image_x = -points_cam_norm[1] * cam1_K[0, 0] + cam1_K[0, 2]
    # 3D Z axis = up = -image y,  iy = -Z*fy + cy
    image_y = -points_cam_norm[2] * cam1_K[1, 1] + cam1_K[1, 2]
    image_points = np.stack([image_x, image_y], axis=-1)
    point_diff = np.abs(cp_points[:, 1:] - image_points)
    point_diff_large = point_diff[(point_diff[:, 0] > 10) | (point_diff[:, 1] > 10)]
    print("point_diff_large", point_diff_large.shape)

    # undistort projected image points
    imsize = (width, height)
    distortion = np.array(intrin[4:])
    print("distortion", distortion)
    mapx, mapy = cv2.initUndistortRectifyMap(cam1_K, distortion, None, cam1_K, imsize, cv2.CV_32F)
    print("map:", mapx.shape, mapy.shape)
    print("mapx[100:-1:200, 100:-1:200]\n", mapx[100:-1:200, 100:-1:200])
    print("mapy[100:-1:200, 100:-1:200]\n", mapy[100:-1:200, 100:-1:200])
    image_y[(image_y < 0) | (image_y > imsize[1] - 1)] = 0
    image_x[(image_x < 0) | (image_x > imsize[0] - 1)] = 0
    image_y = (image_y + 0.5).astype(np.int32)
    image_x = (image_x + 0.5).astype(np.int32)
    undist_image_x = mapx[image_y, image_x]
    undist_image_y = mapy[image_y, image_x]
    undist_image_points = np.stack([undist_image_x, undist_image_y], axis=-1)
    print("conccat shapes:", cp_points.shape, image_points.shape, undist_image_points.shape)
    compare_points = np.concatenate([cp_points[:, 1:], image_points, undist_image_points], axis=1)
    print("compare_points: cp_points, directly projected points, undistorted points\n",
          compare_points[0:-1:1000])

    col_ind = cp_points[:, 1]
    row_ind = cp_points[:, 2]
    imshape = (height, width)
    depth_map = sparse.coo_matrix((points_depth, (row_ind, col_ind)), imshape)
    depth_map = depth_map.toarray()
    return depth_map


def test_synthesize_image():
    set_configs()
    dataset = get_dataset()
    bef_image = None
    bef_pose = tf.identity(4)
    pose_cam2veh = tf.constant([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=tf.float32)
    key = 0
    photo_loss1 = []
    photo_loss2 = []

    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        image = tf.image.decode_jpeg(frame.images[IM_ID].image)
        # rgb to bgr
        image = tf.stack([image[:, :, 2], image[:, :, 1], image[:, :, 0]], axis=-1)
        pose = tf.reshape(frame.images[IM_ID].pose.transform, (4, 4))

        height, width, _ = image.get_shape()
        srcshape = (height, width)
        dstshape = (height // 5, width // 5)
        intrin = frame.context.camera_calibrations[0].intrinsic
        intrin = tf.constant([[intrin[0], 0, intrin[2]], [0, intrin[1], intrin[3]], [0, 0, 1]], dtype=tf.float32)
        depth_map1 = get_depth_map_manually_project(frame, srcshape, dstshape, intrin)
        depth_map2 = get_depth_map_use_cp(frame, srcshape, dstshape, intrin)

        if bef_image is not None:
            relpose = tf.linalg.inv(pose_cam2veh) @ tf.linalg.inv(bef_pose) @ pose @ pose_cam2veh
            print("relative pose from current to before", relpose)
            target_image = tf.image.resize(image, dstshape)

            depth_tensor = tf.constant(depth_map1, dtype=tf.float32)[..., tf.newaxis]
            synthesized = synthesize_image(depth_tensor, bef_image, intrin, relpose)
            synthesized = synthesized[0, 0].numpy()
            photo_error = tf.abs(target_image - synthesized)[depth_tensor[:, :, 0] > 0]
            photo_loss1.append(tf.reduce_mean(photo_error).numpy())
            view = make_view(image.numpy(), depth_map1, dstshape, [bef_image.numpy(), synthesized])
            cv2.imshow("synthesized1", view)

            depth_tensor = tf.constant(depth_map2, dtype=tf.float32)[..., tf.newaxis]
            synthesized = synthesize_image(depth_tensor, bef_image, intrin, relpose)
            synthesized = synthesized[0, 0].numpy()
            photo_error = tf.abs(target_image - synthesized)[depth_tensor[:, :, 0] > 0]
            photo_loss2.append(tf.reduce_mean(photo_error).numpy())
            view = make_view(image.numpy(), depth_map2, dstshape, [bef_image.numpy(), synthesized])
            cv2.imshow("synthesized2", view)
            key = cv2.waitKey(10)
            print("compare loss:", photo_loss1[-1], photo_loss2[-1])

        bef_image = tf.reshape(image, image.get_shape())
        bef_pose = tf.reshape(pose, pose.get_shape())
        if key == ord('q'):
            break

    photo_loss1 = np.array(photo_loss1)
    photo_loss2 = np.array(photo_loss2)
    print("photo_loss1", np.mean(photo_loss1), np.std(photo_loss1))
    print("photo_loss2", np.mean(photo_loss2), np.std(photo_loss2))


def get_depth_map_manually_project(frame, srcshape_hw, dstshape_hw, intrinsic):
    (range_images, camera_projections, range_image_top_pose) = \
        frame_utils.parse_range_image_and_camera_projection(frame)

    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose)

    # xyz points in vehicle frame
    points_veh = np.concatenate(points, axis=0)
    # cp_points: (Nx6) [cam_id, ix, iy, cam_id, ix, iy]
    cp_points = np.concatenate(cp_points, axis=0)[:, :3]
    print("points all:", points_veh.shape, "cp_points", cp_points.shape)

    # extract LiDAR points projected to camera[IM_ID]
    camera_mask = np.equal(cp_points[:, 0], frame.images[IM_ID].name)
    points_veh = points_veh[camera_mask]
    cp_points = cp_points[camera_mask, 1:3]
    print("cam1 points all:", points_veh.shape, "cam1 cp_points", cp_points.shape)

    # transform points from vehicle to camera1
    cam1_T_C2V = tf.reshape(frame.context.camera_calibrations[0].extrinsic.transform, (4, 4)).numpy()
    cam1_T_V2C = np.linalg.inv(cam1_T_C2V)
    points_veh_homo = np.concatenate((points_veh, np.ones((points_veh.shape[0], 1))), axis=1)
    points_veh_homo = points_veh_homo.T
    points_cam_homo = cam1_T_V2C @ points_veh_homo
    points_depth = points_cam_homo[0]

    # project points into image
    # normalize depth to 1
    points_cam = points_cam_homo[:3]
    points_cam_norm = points_cam / points_cam[0:1]
    intrin_np = intrinsic.numpy()
    # scale intrinsic parameters
    scale_y, scale_x = (dstshape_hw[0] / srcshape_hw[0], dstshape_hw[1] / srcshape_hw[1])
    # 3D Y axis = left = -image x,  ix = -Y*fx + cx
    image_x = -points_cam_norm[1] * intrin_np[0, 0] * scale_x + intrin_np[0, 2] * scale_x
    # 3D Z axis = up = -image y,  iy = -Z*fy + cy
    image_y = -points_cam_norm[2] * intrin_np[1, 1] * scale_y + intrin_np[1, 2] * scale_y

    # extract pixels in valid range
    valid_mask = (image_x >= 0) & (image_x <= dstshape_hw[1] - 1) & (image_y >= 0) & (image_y <= dstshape_hw[0] - 1)
    image_x = image_x[valid_mask].astype(np.int32)
    image_y = image_y[valid_mask].astype(np.int32)
    points_depth = points_depth[valid_mask]
    print("points_depth:", points_depth.shape)

    # reconstruct depth map
    depth_map = sparse.coo_matrix((points_depth, (image_y, image_x)), dstshape_hw)
    depth_map = depth_map.toarray()
    return depth_map


def get_depth_map_use_cp(frame, srcshape_hw, dstshape_hw, intrinsic):
    (range_images, camera_projections, range_image_top_pose) = \
        frame_utils.parse_range_image_and_camera_projection(frame)

    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose)

    # xyz points in vehicle frame
    points_veh = np.concatenate(points, axis=0)
    # cp_points: (Nx6) [cam_id, ix, iy, cam_id, ix, iy]
    cp_points = np.concatenate(cp_points, axis=0)[:, :3]
    print("points all:", points_veh.shape, "cp_points", cp_points.shape)

    # extract LiDAR points projected to camera[IM_ID]
    camera_mask = np.equal(cp_points[:, 0], frame.images[IM_ID].name)
    points_veh = points_veh[camera_mask]
    cp_points = cp_points[camera_mask, 1:3]
    print("cam1 points all:", points_veh.shape, "cam1 cp_points", cp_points.shape)

    # transform points from vehicle to camera1
    cam1_T_C2V = tf.reshape(frame.context.camera_calibrations[0].extrinsic.transform, (4, 4)).numpy()
    cam1_T_V2C = np.linalg.inv(cam1_T_C2V)
    points_veh_homo = np.concatenate((points_veh, np.ones((points_veh.shape[0], 1))), axis=1)
    points_veh_homo = points_veh_homo.T
    points_cam_homo = cam1_T_V2C @ points_veh_homo
    points_depth = points_cam_homo[0]

    # scale parameters
    scale_y, scale_x = (dstshape_hw[0] / srcshape_hw[0], dstshape_hw[1] / srcshape_hw[1])
    image_x = cp_points[:, 0] * scale_x
    image_y = cp_points[:, 1] * scale_y
    # extract pixels in valid range
    valid_mask = (image_x >= 0) & (image_x <= dstshape_hw[1] - 1) & (image_y >= 0) & (image_y <= dstshape_hw[0] - 1)
    image_x = image_x[valid_mask].astype(np.int32)
    image_y = image_y[valid_mask].astype(np.int32)
    points_depth = points_depth[valid_mask]

    # reconstruct depth map
    depth_map = sparse.coo_matrix((points_depth, (image_y, image_x)), dstshape_hw)
    depth_map = depth_map.toarray()
    return depth_map


def make_view(image, depth_map, dstshape, cat_images=None):
    dstimsize = (dstshape[1], dstshape[0])
    image = cv2.resize(image, dstimsize)

    depth_img = np.clip(depth_map, 0., 50.) / 50. * 255.
    depth_img = depth_img[..., np.newaxis].astype(np.uint8)
    depth_rgb = cv2.cvtColor(depth_img, cv2.COLOR_GRAY2BGR)
    depth_rgb[(0 < depth_map) & (depth_map < 20), :] = (255, 0, 0)
    depth_rgb[(20 < depth_map) & (depth_map < 40), :] = (0, 255, 0)
    depth_rgb[depth_map > 40, :] = (0, 0, 255)
    depth_rgb = cv2.resize(depth_rgb, dstimsize, cv2.INTER_NEAREST)

    cat_images = [] if cat_images is None else cat_images
    view = [image, depth_rgb]
    for cimg in cat_images:
        cimg = cv2.resize(cimg, dstimsize)
        view.append(cimg)
    view = np.concatenate(view, axis=0)
    return view


def synthesize_image(depth, bimage, intrinsic, pose):
    height, width, _ = bimage.get_shape()
    depth = depth[tf.newaxis, ...]
    bimage = bimage[tf.newaxis, tf.newaxis, ...]
    intrinsic = intrinsic[tf.newaxis, ...]
    pose = pose[tf.newaxis, tf.newaxis, ...]
    bimage = uf.to_float_image(bimage)
    synthesized = SynthesizeSingleScale((1, height, width), 1, 1.)(bimage, intrinsic, depth, pose)
    print("image minmax", tf.reduce_max(bimage), tf.reduce_max(synthesized))
    synthesized = uf.to_uint8_image(synthesized)
    return synthesized


def show_frame_structure():
    dataset = get_dataset()
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        analyze_structure(frame)
        break


def analyze_structure(data, path="frame", space="", depth=0):
    if isinstance(data, bool) or isinstance(data, int) or isinstance(data, float) or (data is None):
        print(f"{space}->{path} = {data}"[:200])
        return

    if isinstance(data, str) or isinstance(data, bytes) or isinstance(data, bytearray):
        print(f"{space}->{path} = {data[:200]}"[:200])
        return

    if depth > 7:
        # print(f"{space}->exceed depth){path}: {data}"[:200])
        return

    print(space + f"[{path}]")

    if isinstance(data, list):
        if data:
            print(f"{space}->list type){path}: len={len(data)}"[:200])
            analyze_structure(data[0], path + "[0]", space + "  ", depth + 1)
        else:
            # print(f"{space}->empty list){path}: {data}"[:200])
            pass
        return

    if isinstance(data, dict):
        if data:
            print(f"{space}->dict type){path}: keys={data.keys()}"[:200])
            for key in data:
                analyze_structure(data[key], path + f"[{key}]", space + "  ", depth + 1)
        else:
            # print(f"{space}->empty dict){path}: {data}"[:200])
            pass
        return

    if "__getitem__" in dir(data):
        if not data:
            return

        try:
            data0 = data[0]
            print(f"{space}->list like){path}: len={len(data)}"[:200])
            analyze_structure(data0, path + "[0]", space + "  ", depth + 1)
            return
        except KeyError as ke:
            pass
        except IndexError as ie:
            pass

    # find attributes of data
    attributes = [var for var in dir(data)]
    variables = []
    for attrib in attributes:
        try:
            if callable(eval(f"data.{attrib}")) or attrib.startswith("__"):
                pass
            elif attrib in ["DESCRIPTOR", "_extensions_by_name", "_extensions_by_number", "_enum_type"]:
                pass
            else:
                variables.append(attrib)
        except AttributeError as ae:
            pass

    if not variables:
        # print(f"{space}{path} has NO variable: type={type(data)} data={data}"[:200])
        return

    print(f"{space}{path} has variables:", variables)
    for varname in variables:
        subdata = eval(f"data.{varname}")
        analyze_structure(subdata, f"{path}.{varname}", space + "  ", depth + 1)


def visualize_range_images():
    dataset = get_dataset()
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        (range_images, camera_projections, range_image_top_pose) = \
            frame_utils.parse_range_image_and_camera_projection(frame)

        plt.figure(figsize=(25, 20))
        for index, image in enumerate(frame.images):
            print("===== show image", index)
            show_labeled_camera_image(image, frame.camera_labels, [3, 3, index + 1])
        plt.show()

        plt.figure(figsize=(64, 20))
        frame.lasers.sort(key=lambda laser: laser.name)
        show_range_image(get_range_image(range_images, open_dataset.LaserName.TOP, 0), 1)
        show_range_image(get_range_image(range_images, open_dataset.LaserName.TOP, 1), 4)
        plt.show()
        break


def show_labeled_camera_image(camera_image, all_camera_labels, layout, cmap=None):
    """Show a camera image and the given camera labels."""
    ax = plt.subplot(*layout)

    # Draw the camera labels.
    for one_camera_labels in all_camera_labels:
        print("camera label:", one_camera_labels.name, camera_image.name)
        # Ignore camera labels that do not correspond to this camera.
        if one_camera_labels.name != camera_image.name:
            continue

        # Iterate over the individual labels.
        for label in one_camera_labels.labels:
            # Draw the object bounding box.
            ax.add_patch(patches.Rectangle(
                            xy=(label.box.center_x - 0.5 * label.box.length,
                                label.box.center_y - 0.5 * label.box.width),
                            width=label.box.length,
                            height=label.box.width,
                            linewidth=1,
                            edgecolor='red',
                            facecolor='none')
                         )

        # Show the camera image.
        plt.imshow(tf.image.decode_jpeg(camera_image.image), cmap=cmap)
        plt.title(open_dataset.CameraName.Name.Name(camera_image.name))
        plt.grid(False)
        plt.axis('off')


def get_range_image(range_images, laser_name, return_index):
    """Returns range image given a laser name and its return index."""
    return range_images[laser_name][return_index]


def show_range_image(range_image, layout_index_start=1):
    """Shows range image.

     Args:
       range_image: the range image data from a given lidar of type MatrixFloat.
       layout_index_start: layout offset
     """
    range_image_tensor = tf.convert_to_tensor(range_image.data)
    range_image_tensor = tf.reshape(range_image_tensor, range_image.shape.dims)
    print("range image shape:", range_image_tensor.get_shape())
    lidar_image_mask = tf.greater_equal(range_image_tensor, 0)
    range_image_tensor = tf.where(lidar_image_mask, range_image_tensor,
                                  tf.ones_like(range_image_tensor) * 1e10)
    range_image_range = range_image_tensor[..., 0]
    range_image_intensity = range_image_tensor[..., 1]
    range_image_elongation = range_image_tensor[..., 2]
    plot_range_image_helper(range_image_range.numpy(), 'range',
                            [8, 1, layout_index_start], vmax=75, cmap='gray')
    plot_range_image_helper(range_image_intensity.numpy(), 'intensity',
                            [8, 1, layout_index_start + 1], vmax=1.5, cmap='gray')
    plot_range_image_helper(range_image_elongation.numpy(), 'elongation',
                            [8, 1, layout_index_start + 2], vmax=1.5, cmap='gray')


def plot_range_image_helper(data, name, layout, vmin=0, vmax=1, cmap='gray'):
    """Plots range image.
    Args:
    data: range image data
    name: the image title
    layout: plt layout
    vmin: minimum value of the passed data
    vmax: maximum value of the passed data
    cmap: color map
    """
    plt.subplot(*layout)
    plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(name)
    plt.grid(False)
    plt.axis('off')


def visualize_camera_projection():
    dataset = get_dataset()
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        (range_images, camera_projections, range_image_top_pose) = \
            frame_utils.parse_range_image_and_camera_projection(frame)

        print("\n===== analyze structure of range_images")
        analyze_structure(range_images, "range_images")
        print("\n===== analyze structure of camera_projections")
        analyze_structure(camera_projections, "camera_projections")
        print("\n===== analyze structure of range_image_top_pose")
        analyze_structure(range_image_top_pose, "range_image_top_pose")

        """
        points[i]: LiDAR i의 xyz 좌표들 [N, 3] 
        cp_points[i]: LiDAR i를 camera에 projection한 이미지 좌표들 [camidx1, iy1, ix1, camidx2, iy2, ix2]
        """
        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose)
        points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
            ri_index=1)

        # 3d points in vehicle frame.
        points_all = np.concatenate(points, axis=0)
        points_all_ri2 = np.concatenate(points_ri2, axis=0)
        # camera projection corresponding to each point.
        cp_points_all = np.concatenate(cp_points, axis=0)
        cp_points_all_ri2 = np.concatenate(cp_points_ri2, axis=0)

        print("===== print points shape ri=0")
        print("points_all", points_all.shape, points_all.dtype)
        print("cp_points_all", cp_points_all.shape, cp_points_all.dtype)
        print("cp_points_all min max", tf.reduce_min(cp_points_all, axis=0).numpy(),
              tf.reduce_max(cp_points_all, axis=0).numpy())
        print("points_all[0:2]\n", points_all[1000:-1:10000])
        print("cp_points_all[0:2]\n", cp_points_all[1500:-1:10000])
        for i in range(5):
            print("  points[i]:", points[i].shape, ", cp_points[i]:", cp_points[i].shape)

        print("===== print points shape ri=1")
        print("points_all_ri2", points_all_ri2.shape)
        print("cp_points_all_ri2\n", cp_points_all_ri2.shape)
        print("points_all_ri2[0:2]\n", points_all_ri2[0:2])
        for i in range(5):
            print("  points_ri2[i]:", points_ri2[i].shape, ", cp_points_ri2[i]:", cp_points_ri2[i].shape)

        images = sorted(frame.images, key=lambda i: i.name)

        print("===== print shapes")
        # The distance between lidar points and vehicle frame origin.
        points_all_dist = tf.norm(points_all, axis=-1, keepdims=True)
        print("points_all_dist", points_all_dist.shape)
        cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)
        print("cp_points_all_tensor", cp_points_all_tensor.shape)

        mask = tf.equal(cp_points_all_tensor[..., 0], images[0].name)
        print("mask shape:", mask.shape, "filter by image name:", images[0].name)
        cp_points_all_tensor = tf.cast(tf.gather_nd(
            cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)
        points_all_dist = tf.gather_nd(points_all_dist, tf.where(mask))

        # projected_points_all_from_raw_data: [ix, iy, dist]
        projected_points_all_from_raw_data = tf.concat(
            [cp_points_all_tensor[..., 1:3], points_all_dist], axis=-1).numpy()

        print("points_all_dist", points_all_dist.shape)
        print("cp_points_all_tensor", cp_points_all_tensor.shape)
        print("projected_points_all_from_raw_data", projected_points_all_from_raw_data.shape)
        plot_points_on_image(projected_points_all_from_raw_data, images[0])
        break


def plot_points_on_image(projected_points, camera_image, point_size=5.0):
    """Plots points on a camera image.
    Args:
    projected_points: [N, 3] numpy array. The inner dims are
      [camera_x, camera_y, range].
    camera_image: jpeg encoded camera image.
    point_size: the point size.

    """
    plot_image(camera_image)
    xs = []
    ys = []
    colors = []
    for point in projected_points:
        xs.append(point[0])  # width, col
        ys.append(point[1])  # height, row
        colors.append(rgba(point[2]))

    plt.scatter(xs, ys, c=colors, s=point_size, edgecolors="none")
    plt.show()


def plot_image(camera_image):
    """Plot a cmaera image."""
    plt.figure(figsize=(20, 12))
    plt.imshow(tf.image.decode_jpeg(camera_image.image))
    plt.grid("off")


def rgba(r):
    """Generates a color based on range.
    Args:
    r: the range value of a given point.
    Returns:
    The color for a given range
    """
    c = plt.get_cmap('jet')((r % 20.0) / 20.0)
    c = list(c)
    c[-1] = 0.5  # alpha
    return c


if __name__ == "__main__":
    # show_front_image_depth_pose()
    test_synthesize_image()
    # show_frame_structure()
    # visualize_range_images()
    # visualize_camera_projection()



