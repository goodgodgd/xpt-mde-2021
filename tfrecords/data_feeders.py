import tensorflow as tf
import cv2
import numpy as np

import settings
from config import opts
from utils.util_class import WrongInputException
import os.path as op
import utils.convert_pose as cp


def feeder_factory(file_list, reader_type, feeder_type="npyfile", stereo=False):
    if feeder_type == "npyfile":
        if stereo and reader_type != "extrinsic":
            FeederClass = NpyFileFeederStereoLeft
        else:
            FeederClass = NpyFileFeeder
    else:
        raise WrongInputException("Wrong feeder type: " + feeder_type)

    if reader_type == "image":
        reader = ImageReaderStereo() if stereo else ImageReader()
    elif reader_type == "intrinsic":
        reader = NpyTxtReaderStereo() if stereo else NpyTxtReader()
    elif reader_type == "depth":
        reader = DepthReaderStereo() if stereo else DepthReader()
    elif reader_type == "pose":
        reader = PoseReaderStereo() if stereo else PoseReader()
    elif reader_type == "extrinsic":
        reader = NpyTxtReader()
    else:
        raise WrongInputException("Wrong reader type: " + reader_type)

    feeder = FeederClass(file_list, reader)
    return feeder


class FeederBase:
    def __init__(self):
        self.parse_type = ""
        self.decode_type = ""
        self.shape = []
    
    def __len__(self):
        raise NotImplementedError()

    def get_next(self):
        raise NotImplementedError()

    def convert_to_feature(self, value):
        raise NotImplementedError()

    def set_type_and_shape(self, value):
        if isinstance(value, np.ndarray):
            if value.dtype == np.uint8:
                self.decode_type = "tf.uint8"
            elif value.dtype == np.float32:
                self.decode_type = "tf.float32"
            else:
                raise WrongInputException(f"[FeederBase] Wrong numpy type: {value.dtype}")
            self.parse_type = "tf.string"
            self.shape = list(value.shape)

        elif isinstance(value, int):
            self.parse_type = "tf.int64"
            self.shape = None

        else:
            raise WrongInputException(f"[FeederBase] Wrong type: {type(value)}")

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


class NpyFeeder(FeederBase):
    def __init__(self):
        super().__init__()

    def convert_to_feature(self, value):
        return self._bytes_feature(value.tostring())


class NpyFileFeeder(NpyFeeder):
    def __init__(self, file_list, file_reader):
        super().__init__()
        self.files = file_list
        self.file_reader = file_reader
        self.idx = -1
        value = self.file_reader(self.files[0])
        self.set_type_and_shape(value[0])

    def __len__(self):
        return len(self.files)

    def get_next(self):
        self.idx = self.idx + 1
        assert self.idx < len(self), f"[FileFeeder] index error: {self.idx} >= {len(self)}"
        data = self.file_reader(self.files[self.idx])
        left = data[0]
        return self.convert_to_feature(left)


class NpyFileFeederStereoLeft(NpyFileFeeder):
    def __init__(self, file_list, file_reader):
        super().__init__(file_list, file_reader)
        self.right_data = None

    def get_next(self):
        self.idx = self.idx + 1
        assert self.idx < len(self), f"[FileFeeder] index error: {self.idx} >= {len(self)}"
        left, right = self.file_reader(self.files[self.idx])
        self.right_data = right
        return self.convert_to_feature(left)


class NpyFileFeederStereoRight(NpyFileFeeder):
    def __init__(self, left_feeder):
        super().__init__(left_feeder.files, left_feeder.file_reader)
        self.left_feeder = left_feeder

    def get_next(self):
        self.idx = self.idx + 1
        assert self.idx < len(self), f"[FileFeeder] index error: {self.idx} >= {len(self)}"
        # right feeder MUST called AFTER calling left feeder
        assert self.left_feeder.idx == self.idx
        return self.convert_to_feature(self.left_feeder.right_data)


class ConstArrayFeeder(NpyFeeder):
    def __init__(self, data, size):
        super().__init__()
        self.data = data
        self.size = size
        self.idx = -1
        self.set_type_and_shape(data)

    def __len__(self):
        return self.size

    def get_next(self):
        self.idx = self.idx + 1
        assert self.idx < len(self), f"[FileFeeder] index error: {self.idx} >= {len(self)}"
        feature = self.convert_to_feature(self.data)
        return feature


# ==================== file readers ====================

class FileReader:
    def __call__(self, filename):
        data = self.read_file(filename)
        splits = self.split_data(data)
        outdata = []
        for split in splits:
            split = self.preprocess(split)
            outdata.append(split)
        return outdata

    def read_file(self, filename):
        raise NotImplementedError()

    def split_data(self, data):
        return [data]

    def preprocess(self, data):
        return data


class ImageReader(FileReader):
    def read_file(self, filename):
        image = cv2.imread(filename)
        return image

    def preprocess(self, image):
        height = opts.IM_HEIGHT
        half_len = int(opts.SNIPPET_LEN // 2)
        # TODO IMPORTANT!
        #   image in 'srcdata': [src--, src-, target, src+, src++]
        #   reordered in 'tfrecords': [src--, src-, src+, src++, target]
        src_up = image[:height * half_len]
        target = image[height * half_len:height * (half_len + 1)]
        src_dw = image[height * (half_len + 1):]
        reordered = np.concatenate([src_up, src_dw, target], axis=0)
        return reordered


class ImageReaderStereo(ImageReader):
    def split_data(self, image):
        width = opts.IM_WIDTH
        left, right = image[:, :width], image[:, width:]
        return [left, right]


class PoseReader(FileReader):
    def read_file(self, filename):
        poses = np.loadtxt(filename)
        half_len = int(opts.SNIPPET_LEN // 2)
        # remove target pose
        poses = np.delete(poses, half_len, 0)
        return poses.astype(np.float32)

    def preprocess(self, poses):
        pose_mats = []
        for pose in poses:
            tmat = cp.pose_quat2matr(pose)
            pose_mats.append(tmat)
        pose_mats = np.stack(pose_mats, axis=0).astype(np.float32)
        return pose_mats


class PoseReaderStereo(PoseReader):
    def split_data(self, poses):
        quat_pose_len = 7
        left, right = poses[:, :quat_pose_len], poses[:, quat_pose_len:]
        return [left, right]


class NpyTxtReader(FileReader):
    def read_file(self, filename):
        data = np.loadtxt(filename)
        return data.astype(np.float32)


class NpyTxtReaderStereo(NpyTxtReader):
    def split_data(self, data):
        intrin_width = 3
        left, right = data[:, :intrin_width], data[:, intrin_width:]
        return [left, right]


class DepthReader(FileReader):
    def read_file(self, filename):
        depth = np.loadtxt(filename)
        # add channel dimension
        depth = np.expand_dims(depth, -1)
        return depth.astype(np.float32)


class DepthReaderStereo(DepthReader):
    def split_data(self, depth):
        width = opts.IM_WIDTH
        left, right = depth[:, :width], depth[:, width:]
        return [left, right]


# ==================== test file readers ====================

def test_image_reader():
    filename = op.join(opts.DATAPATH_SRC, "kitti_raw_train", "2011_09_26_0001", "000024.png")
    original = cv2.imread(filename)
    data = ImageReader()(filename)
    reordered = data[0]
    assert (original.shape == reordered.shape)
    cv2.imshow("original", original)
    cv2.imshow("reordered", reordered)
    cv2.waitKey()
    cv2.destroyAllWindows()


def test_image_reader_stereo():
    filename = op.join(opts.DATAPATH_SRC, "kitti_raw_train", "2011_09_26_0001", "000024.png")
    original = cv2.imread(filename)
    left, right = ImageReaderStereo()(filename)
    assert (original.shape[1] // 2 == left.shape[1]) and (original.shape[1] // 2 == right.shape[1])
    cv2.imshow("original", original)
    cv2.imshow("left", left)
    cv2.imshow("right", right)
    cv2.waitKey()
    cv2.destroyAllWindows()


def test_pose_reader():
    filename = op.join(opts.DATAPATH_SRC, "kitti_raw_train", "2011_09_26_0001", "pose", "000040.txt")
    pose_quat = np.loadtxt(filename)
    pose_mat = PoseReaderStereo()(filename)
    print("quaternion pose:", pose_quat[0])
    print("matrix pose:\n", pose_mat[0])
    assert pose_mat[0].shape == (4, 4, 4)


def test_pose_reader_stereo():
    filename = op.join(opts.DATAPATH_SRC, "kitti_raw_train", "2011_09_26_0001", "pose", "000040.txt")
    pose_quat = np.loadtxt(filename)
    pose_left, pose_right = PoseReaderStereo()(filename)
    print("quaternion pose:", pose_quat[0])
    print("matrix pose:\n", pose_left)
    assert (pose_left.shape == (4, 4, 4)) and (pose_right.shape == pose_left.shape)


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    test_image_reader()
    test_image_reader_stereo()
    test_pose_reader()
    test_pose_reader_stereo()
