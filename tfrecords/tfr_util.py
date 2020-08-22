import numpy as np
import cv2
import tensorflow as tf
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
        config[key] = read_data_config(value)
    return config


def read_data_config(value):
    parse_type = ""
    decode_type = ""
    shape = ()
    if isinstance(value, np.ndarray):
        if value.dtype == np.uint8:
            decode_type = "tf.uint8"
        elif value.dtype == np.float32:
            decode_type = "tf.float32"
        else:
            assert 0, f"[read_data_config] Wrong numpy type: {value.dtype}"
        parse_type = "tf.string"
        shape = list(value.shape)
    elif isinstance(value, int):
        parse_type = "tf.int64"
        shape = None
    else:
        assert 0, f"[read_data_config] Wrong type: {type(value)}"

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


def apply_color_map(depth):
    if len(depth.shape) > 2:
        depth = depth[:, :, 0]
    depth_view = (np.clip(depth, 0, 50.) / 50. * 255).astype(np.uint8)
    depth_view = cv2.applyColorMap(depth_view, cv2.COLORMAP_SUMMER)
    depth_view[depth == 0, :] = (0, 0, 0)
    return depth_view


def show_example(example, wait=0, print_param=False):
    image = example["image"]
    dstshape = (int(image.shape[1] * 1000. / image.shape[0]), 1000)
    image_view = cv2.resize(image, dstshape) if image.shape[0] > 1000 else image.copy()
    cv2.imshow("image", image_view)

    if "image_R" in example:
        image = example["image_R"]
        image_view = cv2.resize(image, dstshape) if image.shape[0] > 1000 else image.copy()
        cv2.imshow("image_R", image_view)

    if "depth_gt" in example:
        depth = example["depth_gt"]
        depth_view = (np.clip(depth, 0, 50.) / 50. * 256).astype(np.uint8)
        depth_view = cv2.applyColorMap(depth_view, cv2.COLORMAP_SUMMER)
        cv2.imshow("depth", depth_view)

    if print_param:
        print("\nintrinsic:\n", example["intrinsic"])
        if "pose_gt" in example:
            print("pose\n", pose_matr2rvec(example["pose_gt"]))

    cv2.waitKey(wait)


