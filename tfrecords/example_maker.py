import numpy as np
import cv2

from tfrecords.readers.waymo_reader import WaymoReader
from tfrecords.readers.city_reader import CityReader


class ExampleMaker:
    def __init__(self, dataset, split, shwc_shape, data_keys, reader_args=None):
        self.dataset = dataset
        self.split = split
        self.shwc_shape = shwc_shape
        self.data_keys = data_keys
        self.data_reader = WaymoReader()
        self.reader_args = reader_args

    def init_reader(self, drive_path):
        self.data_reader = self.data_reader_factory()
        self.data_reader.init_drive(drive_path)

    def data_reader_factory(self):
        if self.dataset.startswith("cityscapes"):
            return CityReader(self.split, self.reader_args)     # split and ZipFile object
        if self.dataset is "waymo":
            return WaymoReader(self.split)
        else:
            assert 0, f"[data_reader_factory] invalid dataset name {self.dataset}"

    def num_frames(self):
        return self.data_reader.num_frames_()

    def get_range(self):
        return self.data_reader.get_range_()

    def get_example(self, index):
        frame_id, frame_seq_ids = self.make_snippet_ids(index)
        example = dict()
        example["image"], raw_shape_hwc = self.load_snippet_images(frame_seq_ids)
        example["intrinsic"] = self.load_intrinsic(frame_id, raw_shape_hwc)
        if "depth_gt" in self.data_keys:
            example["depth_gt"] = self.load_depth_map(frame_id, raw_shape_hwc)
        if "pose_gt" in self.data_keys:
            example["pose_gt"] = self.load_snippet_poses(frame_seq_ids)
        if "image_R" in self.data_keys:
            example["image_R"], _ = self.load_snippet_images(frame_seq_ids, right=True)
        if "intrinsic_R" in self.data_keys:
            example["intrinsic_R"] = self.load_intrinsic(frame_id, raw_shape_hwc, right=True)
        if "pose_gt_R" in self.data_keys:
            example["pose_gt_R"] = self.load_snippet_poses(frame_seq_ids, right=True)
        if "stereo_T_LR" in self.data_keys:
            example["stereo_T_LR"] = self.data_reader.get_stereo_extrinsic(frame_id)

        if index % 100 == 10:
            self.show_example(example, 200)
        if index % 500 == 10:
            print("\nintrinsic:\n", example["intrinsic"])
            if "pose_gt" in example:
                print("pose\n", example["pose_gt"])

        example = self.verify_snippet(example)
        return example

    def make_snippet_ids(self, frame_index):
        frame_id = self.data_reader.index_to_id(frame_index)
        halflen = self.shwc_shape[0] // 2
        max_frame_index = self.num_frames() - 1
        frame_seq_ids = np.arange(frame_id-halflen, frame_id+halflen+1)
        frame_seq_ids = np.clip(frame_seq_ids, 0, max_frame_index).tolist()
        return frame_id, frame_seq_ids

    def load_snippet_images(self, frame_ids, right=False):
        image_seq = []
        raw_shape = self.shwc_shape[1:]
        for fid in frame_ids:
            image = self.data_reader.get_image(fid, right=right)
            if image is None:
                return None
            raw_shape = image.shape
            dstsize_wh = (self.shwc_shape[2], self.shwc_shape[1])
            image = cv2.resize(image, dstsize_wh)
            image_seq.append(image)
        # move target image to the bottom
        target_index = self.shwc_shape[0] // 2
        target_image = image_seq.pop(target_index)
        image_seq.append(target_image)
        image_seq = np.concatenate(image_seq, axis=0).astype(np.uint8)
        return image_seq, raw_shape

    def load_intrinsic(self, index, raw_shape_hwc, right=False):
        intrinsic = self.data_reader.get_intrinsic(index, right=right)
        if intrinsic is None:
            return None
        scale_y = self.shwc_shape[1] / raw_shape_hwc[0]
        scale_x = self.shwc_shape[2] / raw_shape_hwc[1]
        intrinsic[0] = intrinsic[0] * scale_x
        intrinsic[1] = intrinsic[1] * scale_y
        return intrinsic.astype(np.float32)

    def load_snippet_poses(self, frame_ids, right=False):
        pose_seq = []
        for fid in frame_ids:
            pose = self.data_reader.get_pose(fid, right=right)
            if pose is None:
                return None
            pose_seq.append(pose)
        target_index = self.shwc_shape[0] // 2
        target_pose = pose_seq.pop(target_index)
        pose_seq = [np.linalg.inv(pose) @ target_pose for pose in pose_seq]
        pose_seq = np.stack(pose_seq, axis=0)
        return pose_seq.astype(np.float32)

    def load_depth_map(self, index, raw_shape_hwc):
        intrinsic = self.data_reader.get_intrinsic(index)
        depth_map = self.data_reader.get_depth(index, raw_shape_hwc[:2], self.shwc_shape[1:3], intrinsic)
        return depth_map.astype(np.float32)

    def show_example(self, example, wait=0):
        image = example["image"]
        image_view = cv2.resize(image, (int(image.shape[1] * 1000. / image.shape[0]), 1000))
        depth = example["depth_gt"]
        depth_view = (np.clip(depth, 0, 50.) / 50. * 256).astype(np.uint8)
        depth_view = cv2.applyColorMap(depth_view, cv2.COLORMAP_SUMMER)
        cv2.imshow("image", image_view)
        cv2.imshow("depth", depth_view)
        cv2.waitKey(wait)
        # print("\nintrinsic:\n", example["intrinsic"])
        # if "pose_gt" in example:
        #     print("pose\n", example["pose_gt"])

    def verify_snippet(self, example):
        if self.dataset is "waymo":
            poses = example["pose_gt"]
            positions = poses[:, :3, 3]
            distances = np.linalg.norm(positions, axis=1)

            min_dist = np.min(distances)
            if min_dist < 0.2:
                return dict()   # empty dict means skip this frame

            max_dist = np.max(distances)
            if max_dist > 5.:
                print("\n  Change scene? distance=", max_dist)
                return dict()   # empty dict means skip this frame
        return example

