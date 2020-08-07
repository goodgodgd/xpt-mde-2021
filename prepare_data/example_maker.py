import cv2
import numpy as np

import settings
from config import opts
import prepare_data.readers.kitti_reader as kr
import prepare_data.readers.city_reader as cr
import utils.convert_pose as cp
from utils.util_class import WrongInputException


class ExampleMaker:
    def __init__(self, base_path, snippet_len, pose_avail, depth_avail, stereo=False):
        self.base_path = base_path
        self.pose_avail = pose_avail
        self.depth_avail = depth_avail
        self.snippet_len = snippet_len
        self.data_reader = kr.KittiRawTrainReader("", "", True)
        self.drive_path = ""
        self.num_frames = 0
        self.stereo = stereo

    def set_reader(self, reader):
        assert reader.stereo is False
        self.data_reader = reader
        self.num_frames = len(reader.frame_names)

    def get_example(self, example_index):
        snippet_frame_inds = self.make_snippet_indices(example_index)
        example = dict()
        example["index"] = example_index
        example["image"], raw_img_shape = self.load_snippet_frames(snippet_frame_inds)
        example["intrinsic"] = self.load_intrinsic(raw_img_shape)
        if self.pose_avail:
            example["pose_gt"] = self.load_snippet_poses(snippet_frame_inds)
        if self.depth_avail:
            example["depth_gt"] = self.load_frame_depth(example_index, raw_img_shape)
        if self.stereo:
            example["stereo_T_LR"] = self.data_reader.get_stereo_extrinsic()
        return example

    def make_snippet_indices(self, frame_idx):
        halflen = self.snippet_len // 2
        max_frame_index = self.data_reader.num_frames() - 1
        indices = np.arange(frame_idx-halflen, frame_idx+halflen+1)
        indices = np.clip(indices, 0, max_frame_index).tolist()
        return indices

    def load_snippet_frames(self, snippet_ids):
        frames = []
        raw_img_shape = ()
        for idnum in snippet_ids:
            frame = self.data_reader.get_image(idnum)
            raw_img_shape = frame.shape[:2]
            frame = cv2.resize(frame, dsize=opts.get_shape("WH"), interpolation=cv2.INTER_LINEAR)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frames.append(frame)

        frames = np.concatenate(frames, axis=0)
        return frames, raw_img_shape

    def load_snippet_poses(self, snippet_ids):
        poses = []
        for idnum in snippet_ids:
            pose = self.data_reader.get_quat_pose(idnum)
            poses.append(pose)

        poses = np.stack(poses, axis=0)
        poses = self.to_local_pose(poses, self.snippet_len // 2)
        return poses

    def to_local_pose(self, poses, target_index):
        tgt_to_src_poses = []
        target_pose_mat = cp.pose_quat2matr(poses[target_index])
        for pose in poses:
            cur_pose_mat = cp.pose_quat2matr(pose)
            tgt_to_src_mat = np.matmul(np.linalg.inv(cur_pose_mat), target_pose_mat)
            tgt_to_src_qpose = cp.pose_matr2quat(tgt_to_src_mat)
            tgt_to_src_poses.append(tgt_to_src_qpose)

        tgt_to_src_poses = np.stack(tgt_to_src_poses, axis=0)
        return tgt_to_src_poses

    def load_frame_depth(self, frame_idx, raw_img_shape):
        dst_shape = opts.get_shape("HW")
        depth_map = self.data_reader.get_depth_map(frame_idx, raw_img_shape, dst_shape)
        return depth_map

    def load_intrinsic(self, raw_img_shape):
        intrinsic = self.data_reader.get_intrinsic()
        sx = opts.get_shape("W") / raw_img_shape[1]
        sy = opts.get_shape("H") / raw_img_shape[0]
        out = intrinsic.copy()
        out[0, 0] *= sx
        out[0, 2] *= sx
        out[1, 1] *= sy
        out[1, 2] *= sy
        return out


class ExampleMakerStereo(ExampleMaker):
    def __init__(self, base_path, snippet_len, pose_avail, depth_avail):
        super().__init__(base_path, snippet_len, pose_avail, depth_avail, True)

    def set_reader(self, reader):
        assert reader.stereo
        self.data_reader = reader
        self.num_frames = len(reader.frame_names)

    def load_snippet_frames(self, snippet_ids):
        frames = []
        raw_img_shape = ()
        for idnum in snippet_ids:
            img_lef, img_rig = self.data_reader.get_image(idnum)
            raw_img_shape = img_lef.shape[:2]
            img_lef = cv2.resize(img_lef, opts.get_shape("WH"), interpolation=cv2.INTER_LINEAR)
            img_lef = cv2.cvtColor(img_lef, cv2.COLOR_RGB2BGR)
            img_rig = cv2.resize(img_rig, opts.get_shape("WH"), interpolation=cv2.INTER_LINEAR)
            img_rig = cv2.cvtColor(img_rig, cv2.COLOR_RGB2BGR)
            frame = np.concatenate([img_lef, img_rig], axis=1)
            frames.append(frame)

        frames = np.concatenate(frames, axis=0)
        return frames, raw_img_shape

    def load_intrinsic(self, raw_img_shape):
        intrin_lef, intrin_rig = self.data_reader.get_intrinsic()
        intrinsic = np.concatenate([intrin_lef, intrin_rig], axis=1)
        sx = opts.get_shape("W") / raw_img_shape[1]
        sy = opts.get_shape("H") / raw_img_shape[0]
        intrinsic[0, :] = intrinsic[0, :] * sx
        intrinsic[1, :] = intrinsic[1, :] * sy
        return intrinsic

    def load_snippet_poses(self, snippet_ids):
        pose_seq_lef = []
        pose_seq_rig = []
        for ind in snippet_ids:
            pose_lef, pose_rig = self.data_reader.get_quat_pose(ind)
            pose_seq_lef.append(pose_lef)
            pose_seq_rig.append(pose_rig)

        pose_seq_lef = np.stack(pose_seq_lef, axis=0)
        pose_seq_rig = np.stack(pose_seq_rig, axis=0)
        pose_seq_lef = self.to_local_pose(pose_seq_lef, self.snippet_len // 2)
        pose_seq_rig = self.to_local_pose(pose_seq_rig, self.snippet_len // 2)
        poses = np.concatenate([pose_seq_lef, pose_seq_rig], axis=1)
        return poses

    def load_frame_depth(self, frame_idx, raw_img_shape):
        dst_shape = opts.get_shape("HW")
        depth_lef, depth_rig = self.data_reader.get_depth_map(frame_idx, raw_img_shape, dst_shape)
        depth = np.concatenate([depth_lef, depth_rig], axis=1)
        return depth


def test_kitti_loader():
    np.set_printoptions(precision=3, suppress=True, linewidth=150)
    dataset = "kitti_raw"
    maker, reader = dataset_loader_factory(get_raw_data_path(dataset), dataset, "train")
    delay = 0
    height, width = opts.get_shape("HW")
    drive_paths = reader.list_drive_paths()

    for drive_path in drive_paths:
        frame_indices = reader.init_drive(drive_path)
        maker.set_reader(reader)
        if len(frame_indices) == 0:
            print("this drive is EMPTY")
            continue

        for index in frame_indices:
            example = maker.get_example(index)
            index = example["index"]
            frames = example["image"]
            poses = example["pose_gt"]
            depth_map = example["depth_gt"]
            intrinsic = example["intrinsic"]
            depth_map = np.clip(depth_map, 0, 20)
            print(f"frame {index}: concatenated image shape={frames.shape}, pose shape={poses.shape}")
            print("pose", poses[:3, :])
            print("intrinsic", intrinsic)

            if opts.STEREO:
                target_lef = frames[height * 2:height * 3, :width, :]
                target_rig = frames[height * 2:height * 3, width:, :]
                frame_view = np.concatenate([target_lef, target_rig], axis=0)
                depth_lef = depth_map[:, :width]
                depth_rig = depth_map[:, width:]
                depth_view = np.concatenate([depth_lef, depth_rig], axis=0)
                cv2.imshow("frame_target", frame_view)
                cv2.imshow("depth_target", depth_view)
            else:
                cv2.imshow("depth_map", depth_map)

            cv2.imshow("frames", frames)
            key = cv2.waitKey(delay)
            if key == ord('q'):
                return
            if key == ord('n'):
                break
            if key == ord('s'):
                if delay > 0:
                    delay = 0
                else:
                    delay = 500


if __name__ == "__main__":
    test_kitti_loader()
