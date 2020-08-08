import numpy as np
from tfrecords.readers.waymo_reader import WaymoReader


class ExampleMaker:
    def __init__(self, dataset, split, shwc_shape, data_keys):
        self.split = split
        self.shwc_shape = shwc_shape
        self.data_keys = data_keys
        self.data_reader = self.get_data_reader(dataset, split)

    def get_data_reader(self, dataset, split):
        if dataset is "waymo":
            return WaymoReader(split)
        else:
            assert 0, f"[get_data_reader] invalid dataset name {dataset}"

    def list_frames_(self, drive_path):
        self.data_reader.init_drive(drive_path)

    def num_frames(self):
        self.data_reader.num_frames()

    def get_example(self, index):
        frame_ids = self.make_snippet_ids(index)
        example = dict()
        example["image"] = self.load_snippet_images(frame_ids)
        example["intrinsic"] = self.data_reader.get_intrinsic(index)
        raw_shape_hwc = example["image"].shape
        if "depth_gt" in self.data_keys:
            example["depth_gt"] = self.data_reader.get_depth(index, raw_shape_hwc[:2], self.shwc_shape[1:3])
        if "pose_gt" in self.data_keys:
            example["pose_gt"] = self.load_snippet_poses(frame_ids)
        if "image_R" in self.data_keys:
            example["image_R"] = self.load_snippet_images(frame_ids, right=True)
        if "intrinsic_R" in self.data_keys:
            example["intrinsic_R"] = self.data_reader.get_intrinsic(index, right=True)
        if "pose_gt_R" in self.data_keys:
            example["pose_gt_R"] = self.load_snippet_poses(frame_ids, right=True)
        if "stereo_T_LR" in self.data_keys:
            example["stereo_T_LR"] = self.data_reader.get_stereo_extrinsic()

    def make_snippet_ids(self, frame_index):
        frame_id = self.data_reader.index_to_id(frame_index)
        halflen = self.shwc_shape[0] // 2
        max_frame_index = self.data_reader.num_frames() - 1
        frame_seq_ids = np.arange(frame_id-halflen, frame_id+halflen+1)
        frame_seq_ids = np.clip(frame_seq_ids, 0, max_frame_index).tolist()
        return frame_seq_ids

    def load_snippet_images(self, frame_ids, right=False):
        image_seq = []
        for fid in frame_ids:
            image = self.data_reader.get_image(fid, right=right)
            image_seq.append(image)
        image_seq = np.concatenate(image_seq, axis=0)
        return image_seq

    def load_snippet_poses(self, frame_ids, right=False):
        pose_seq = []
        for fid in frame_ids:
            pose = self.data_reader.get_pose(fid, right=right)
            pose_seq.append(pose)
        pose_seq = np.stack(pose_seq, axis=0)
        return pose_seq
