import numpy as np
import cv2
import tensorflow as tf
import pandas as pd
from utils.convert_pose import pose_matr2rvec


class Serializer:
    def __call__(self, example_dict):
        features = self.convert_to_feature(example_dict)
        # wrap the data as TensorFlow Features.
        features = tf.train.Features(feature=features)
        # wrap again as a TensorFlow Example.
        example = tf.train.Example(features=features)
        # serialize the data.
        serialized = example.SerializeToString()
        return serialized

    def convert_to_feature(self, example_dict):
        features = dict()
        for key, value in example_dict.items():
            if value is None:
                continue
            elif isinstance(value, np.ndarray):
                features[key] = self._bytes_feature(value.tostring())
            elif isinstance(value, int):
                features[key] = self._int64_feature(value)
            elif isinstance(value, float):
                features[key] = self._int64_feature(value)
            else:
                assert 0, f"[convert_to_feature] Wrong data type: {type(value)}"
        return features

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def inspect_properties(example):
    config = dict()
    for key, value in example.items():
        if value is not None:
            config[key] = read_data_config(key, value)
    return config


def read_data_config(key, value):
    parse_type = ""
    decode_type = ""
    shape = ()
    if isinstance(value, np.ndarray):
        if value.dtype == np.uint8:
            decode_type = "tf.uint8"
        elif value.dtype == np.float32:
            decode_type = "tf.float32"
        else:
            assert 0, f"[read_data_config] Wrong numpy type: {value.dtype}, key={key}"
        parse_type = "tf.string"
        shape = list(value.shape)
    elif isinstance(value, int):
        parse_type = "tf.int64"
        shape = None
    else:
        assert 0, f"[read_data_config] Wrong type: {type(value)}, key={key}"

    return {"parse_type": parse_type, "decode_type": decode_type, "shape": shape}


def resize_depth_map(depth_map, srcshape_hw, dstshape_hw):
    if depth_map.ndim == 3:
        depth_map = depth_map[:, :, 0]
    # depth_view = apply_color_map(depth_map)
    # depth_view = cv2.resize(depth_view, (dstshape_hw[1], dstshape_hw[0]))
    # cv2.imshow("srcdepth", depth_view)
    du, dv = np.meshgrid(np.arange(dstshape_hw[1]), np.arange(dstshape_hw[0]))
    du, dv = (du.reshape(-1), dv.reshape(-1))
    scale_y, scale_x = (srcshape_hw[0] / dstshape_hw[0], srcshape_hw[1] / dstshape_hw[1])
    su, sv = (du * scale_x).astype(np.uint16), (dv * scale_y).astype(np.uint16)
    radi_x, radi_y = (int(scale_x/2), int(scale_y/2))
    # print("su", su[0:800:40])
    # print("sv", sv[0:-1:10000])

    dst_depth = np.zeros(du.shape).astype(np.float32)
    weight = np.zeros(du.shape).astype(np.float32)
    for sdy in range(-radi_y, radi_y+1):
        for sdx in range(-radi_x, radi_x+1):
            v_inds = np.clip(sv + sdy, 0, srcshape_hw[0] - 1).astype(np.uint16)
            u_inds = np.clip(su + sdx, 0, srcshape_hw[1] - 1).astype(np.uint16)

            # if (dx==1) and (dy==1):
            #     print("u_inds", u_inds[0:400:20])
            #     print("v_inds", v_inds[0:-1:10000])
            tmp_depth = depth_map[v_inds, u_inds]
            tmp_weight = (tmp_depth > 0).astype(np.uint8)
            dst_depth += tmp_depth
            weight += tmp_weight

    dst_depth[weight > 0] /= weight[weight > 0]
    dst_depth = dst_depth.reshape((dstshape_hw[0], dstshape_hw[1], 1))
    return dst_depth


def point_cloud_to_depth_map(src_pcd, intrinsic, imshape, T2cam=np.eye(4)):
    """
    :param src_pcd: source point cloud [N, 4]
    :param intrinsic: [3, 3]
    :param imshape: height and width of output depth map
    :param T2cam: transformation matrix to camera frame
    :return: depth map
    """
    if src_pcd.shape[1] == 3:
        src_pcd = np.concatenate([src_pcd, np.ones((1, src_pcd.shape[1]))])
    src_pcd = src_pcd.T    # (N, 4) => (4, N)
    src_pcd[3, :] = 1
    # points in camera frame (x:right, y:down, z:depth) (3, N)
    points = np.dot(T2cam, src_pcd)[:3]
    points = points[:, points[2] > 1.]
    # project to camera, pixels: [3, N]
    pixels = np.dot(intrinsic, points) / points[2:3]
    assert np.isclose(pixels[2], 1.).all()
    # remove pixels out of image plane
    pixels = pixels[:, (pixels[0]>=0) & (pixels[0]<imshape[1]-1) & (pixels[1]>=0) & (pixels[1]<imshape[0]-1)]
    # quarter pixels around `pixels`
    data = np.stack([np.floor(pixels[0]), np.floor(pixels[1]), np.ceil(pixels[0]), np.ceil(pixels[1])], axis=1)
    quart_pixels = pd.DataFrame(data, columns=['x1', 'y1', 'x2', 'y2'])
    quart_pixels = quart_pixels.astype(int)
    quarter_columns = [['x1', 'y1'], ['x1', 'y2'], ['x2', 'y1'], ['x2', 'y2']]
    depthmap = np.zeros(imshape, dtype=np.float32)
    weightmap = np.zeros(imshape, dtype=np.float32)
    flpixels = pixels[:2]

    for quarter_col in quarter_columns:
        qtpixels = quart_pixels.loc[:, quarter_col]
        qtpixels = qtpixels.rename(columns={quarter_col[0]: 'col', quarter_col[1]: 'row'})
        # diff = (1-abs(x-xn), 1-abs(y-yn)) [N, 2]
        diff = 1 - np.abs(flpixels.T - qtpixels.values)
        # weights = (1-abs(x-xn)) * (1-abs(y-yn)) [N]
        weights = diff[:, 0] * diff[:, 1]

        step = 0
        while len(qtpixels.index) > 0:
            step += 1
            step_pixels = qtpixels.drop_duplicates(keep='first')
            rows = step_pixels['row'].values
            cols = step_pixels['col'].values
            inds = step_pixels.index.values
            depthmap[rows, cols] += points[2, inds] * weights[inds]
            weightmap[rows, cols] += weights[inds]
            qtpixels = qtpixels[~qtpixels.index.isin(step_pixels.index)]

    depthmap[depthmap > 0] = depthmap[depthmap > 0] / weightmap[depthmap > 0]
    depthmap[weightmap < 0.5] = 0
    return depthmap


def apply_color_map(depth):
    if len(depth.shape) > 2:
        depth = depth[:, :, 0]
    depth_view = (np.clip(depth, 0, 50.) / 50. * 255).astype(np.uint8)
    depth_view = cv2.applyColorMap(depth_view, cv2.COLORMAP_SUMMER)
    depth_view[depth == 0, :] = (0, 0, 0)
    return depth_view


def show_example(example, wait=0, print_param=False, max_height=1000, suffix=""):
    image = example["image"]
    dstsize_wh = (image.shape[1], image.shape[0])
    if max_height:
        dstsize_wh = (int(image.shape[1] * max_height / image.shape[0]), max_height)
    image_view = cv2.resize(image, dstsize_wh) if image.shape[0] > 1000 else image.copy()
    cv2.imshow("image" + suffix, image_view)

    if "image_R" in example and example["image_R"] is not None:
        image = example["image_R"]
        image_view = cv2.resize(image, dstsize_wh) if image.shape[0] > 1000 else image.copy()
        cv2.imshow("image_R" + suffix, image_view)

    if "depth_gt" in example and example["depth_gt"] is not None:
        depth = example["depth_gt"]
        depth_view = (np.clip(depth, 0, 50.) / 50. * 256).astype(np.uint8)
        depth_view = cv2.applyColorMap(depth_view, cv2.COLORMAP_SUMMER)
        cv2.imshow("depth" + suffix, depth_view)

    if print_param:
        print("\nintrinsic:\n", example["intrinsic"])
        if "pose_gt" in example and example["pose_gt"] is not None:
            print("pose\n", pose_matr2rvec(example["pose_gt"]))

    cv2.waitKey(wait)


# ======================================================================
import pykitti


def test_point_cloud_to_depth_map():
    print("\n===== start test_kitti_odom_reader")
    scale = 2
    drive_loader = pykitti.raw("/media/ian/IanBook2/datasets/kitti_raw_data", "2011_09_26", "0002")
    intrinsic = drive_loader.calib.K_cam2
    intrinsic[:2] = intrinsic[:2] / scale
    for index in range(100):
        velo_data = drive_loader.get_velo(index)
        images = drive_loader.get_rgb(index)
        image = np.array(images[0])
        image = cv2.resize(image, (image.shape[1]//scale, image.shape[0]//scale))
        depthmap = point_cloud_to_depth_map(velo_data, intrinsic,
                                            image.shape[:2], drive_loader.calib.T_cam2_velo)
        depthmap = apply_color_map(depthmap)
        view = np.concatenate([image, depthmap], axis=0)
        cv2.imshow("depth", view)
        cv2.waitKey()
    print("!!! test_point_cloud_to_depth_map passed")


if __name__ == "__main__":
    test_point_cloud_to_depth_map()
