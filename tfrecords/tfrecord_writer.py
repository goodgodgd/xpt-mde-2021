import os.path as op
from glob import glob
import tensorflow as tf
import cv2
import numpy as np
import json

import settings
from config import opts
import utils.util_funcs as uf
import tfrecords.data_feeders as df


class TfrecordMaker:
    def __init__(self, srcpath, dstpath):
        self.srcpath = srcpath
        self.dstpath = dstpath
        # check if there is depth data available
        depths = glob(srcpath + "/*/depth/*.txt")
        self.depth_avail = True if depths else False

    def make(self):
        try:
            data_feeders = self.create_feeders()
        except ValueError as e:
            print(e)
            return

        self.write_tfrecord_config(data_feeders)
        num_images = len(data_feeders["image"])
        num_shards = max(min(num_images // 5000, 10), 1)
        num_images_per_shard = num_images // num_shards
        print(f"========== tfrecord maker started: srcpath={self.srcpath}, dstpath={self.dstpath}")
        print(f"num images={num_images}, shards={num_shards}, images per shard={num_images_per_shard}")
        uf.print_progress(num_images, is_total=True)

        for si in range(num_shards):
            outfile = f"{self.dstpath}/shard_{si:02d}.tfrecord"
            print("\n===== start creating:", outfile)
            with tf.io.TFRecordWriter(outfile) as writer:
                for fi in range(si*num_images_per_shard, (si+1)*num_images_per_shard):
                    uf.print_progress(fi)
                    raw_example = self.create_next_example_dict(data_feeders)
                    serialized = self.make_serialized_example(raw_example)
                    writer.write(serialized)

        print(f"\ntfrecord maker finished: srcpath={self.srcpath}, dstpath={self.dstpath}")

    def create_feeders(self):
        image_files, depth_files, pose_files, intrin_files = self.list_sequence_files()
        if depth_files:
            depth_feeder = df.NpyFeeder(depth_files, depth_reader)
        else:
            zero_depth = np.zeros((opts.IM_HEIGHT, opts.IM_WIDTH, 1), dtype=np.float32)
            depth_feeder = df.ConstArrayFeeder(zero_depth, len(image_files))

        feeders = {"image": df.NpyFeeder(image_files, image_reader),
                   "depth": depth_feeder,
                   "pose": df.NpyFeeder(pose_files, pose_reader),
                   "intrinsic": df.NpyFeeder(intrin_files, txt_reader)
                   }
        return feeders

    def list_sequence_files(self):
        image_files = glob(op.join(self.srcpath, "*/*.png"))
        if not image_files:
            raise ValueError(f"[list_sequence_files] no image file in {self.srcpath}")
        if self.depth_avail:
            depth_files = [op.join(op.dirname(file_path), "depth",
                                   op.basename(file_path).replace(".png", ".txt"))
                           for file_path in image_files]
        else:
            depth_files = []

        pose_files = [op.join(op.dirname(file_path), "pose",
                              op.basename(file_path).replace(".png", ".txt"))
                      for file_path in image_files]
        intrin_files = [op.join(op.dirname(file_path), "intrinsic.txt")
                        for file_path in image_files]

        print("=== list sequence files")
        print(f"frame: {image_files[0:1000:100]}")
        print(f"depth: {depth_files[0:1000:100]}")
        print(f"pose: {pose_files[0:1000:100]}")
        print(f"intrin: {intrin_files[0:1000:100]}")
        
        for files in [image_files, depth_files, pose_files, intrin_files]:
            for file in files:
                assert op.isfile(file), f"{file} NOT exist"

        return image_files, depth_files, pose_files, intrin_files

    def write_tfrecord_config(self, feeders):
        config = dict()
        for key, feeder in feeders.items():
            single_config = {"parse_type": feeder.parse_type, "decode_type": feeder.decode_type, "shape": feeder.shape}
            config[key] = single_config

        print("=== config", config)
        with open(op.join(self.dstpath, "tfr_config.txt"), "w") as fr:
            json.dump(config, fr)

    @staticmethod
    def create_next_example_dict(feeders):
        example = dict()
        for key, feeder in feeders.items():
            example[key] = feeder.get_next()
        return example

    @staticmethod
    def make_serialized_example(data_dict):
        # wrap the data as TensorFlow Features.
        features = tf.train.Features(feature=data_dict)
        # wrap again as a TensorFlow Example.
        example = tf.train.Example(features=features)
        # serialize the data.
        serialized = example.SerializeToString()
        return serialized


# ==================== file readers ====================

def image_reader(filename):
    """
    reorder image: [src0 src1 tgt src2 src3] -> [src1 src2 src3 src4 tgt]
    """
    image = cv2.imread(filename)
    height = int(image.shape[0] // opts.SNIPPET_LEN)
    half_len = int(opts.SNIPPET_LEN // 2)
    src_up = image[:height*half_len]
    target = image[height*half_len:height*(half_len+1)]
    src_dw = image[height*(half_len+1):]
    reordered = np.concatenate([src_up, src_dw, target], axis=0)
    return reordered


def pose_reader(filename):
    """
    quaternion based pose to transformation matrix omitting target pose (identity)
    order: [src0 src1 tgt src2 src3] -> [src1 src2 src3 src4]
    shape: [5, 7] -> [4, 4, 4]
    """
    poses = np.loadtxt(filename)
    half_len = int(opts.SNIPPET_LEN // 2)
    poses = np.delete(poses, half_len, 0)
    pose_mats = []
    for pose in poses:
        tmat = uf.pose_quat2matr(pose)
        pose_mats.append(tmat)
    pose_mats = np.stack(pose_mats, axis=0)
    return pose_mats.astype(np.float32)


def npy_reader(filename):
    data = np.load(filename)
    return data.astype(np.float32)


def txt_reader(filename):
    data = np.loadtxt(filename)
    return data.astype(np.float32)


def depth_reader(filename):
    data = np.loadtxt(filename)
    data = np.expand_dims(data, -1)
    return data.astype(np.float32)


# ==================== test file readers ====================

def test_image_reader():
    filename = op.join(opts.DATAPATH_SRC, "kitti_raw_train", "2011_09_26_0001", "000024.png")
    original = cv2.imread(filename)
    reordered = image_reader(filename)
    assert (original.shape == reordered.shape)
    cv2.imshow("original", original)
    cv2.imshow("reordered", reordered)
    cv2.waitKey()


def test_pose_reader():
    filename = op.join(opts.DATAPATH_SRC, "kitti_raw_train", "2011_09_26_0001", "pose", "000040.txt")
    pose_quat = np.loadtxt(filename)
    pose_tmat = pose_reader(filename)
    print("quaternion pose:", pose_quat[0])
    print("matrix pose:\n", pose_tmat[0])
    assert pose_tmat.shape == (4, 4, 4)


def test():
    np.set_printoptions(precision=3, suppress=True)
    test_image_reader()
    test_pose_reader()


if __name__ == "__main__":
    test()
