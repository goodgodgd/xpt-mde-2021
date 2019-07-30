import os.path as op
from glob import glob
import pykitti


class KittiUtil:
    def __init__(self):
        static_file = op.join(op.dirname(op.abspath(__file__)), "resources", "kitti_static_frames.txt")
        self.static_frames = self.read_static_frames(static_file)

    def read_static_frames(self, filename):
        with open(filename, "r") as fr:
            lines = fr.readlines()
            static_frames = [line.strip("\n") for line in lines]
        return static_frames

    def remove_static_frames(self, frames):
        valid_frames = [frame for frame in frames if frame not in self.static_frames]
        print(f"[remove_static_frames] {len(frames)} -> {len(valid_frames)}")
        return valid_frames

    def list_drives(self, dataset, split):
        filename = op.join(op.dirname(op.abspath(__file__)), "resources", f"{dataset}_{split}_scenes.txt")
        with open(filename, "r") as f:
            drives = f.readlines()
            drives = [tuple(drive.strip("\n").split()) for drive in drives]
            print("drive list:", drives)
            return drives

    def create_drive_loader(self, base_path, drive):
        raise NotImplementedError()

    def frame_indices(self, num_frames, snippet_len):
        raise NotImplementedError()


class KittiRawTrainUtil(KittiUtil):
    def __init__(self):
        super().__init__()

    def create_drive_loader(self, base_path, drive):
        date, drive_id = drive
        return pykitti.raw(base_path, date, drive_id)

    def frame_indices(self, drive_path, snippet_len):
        print("drive path", drive_path)
        # list frame files in drive_path
        frame_pattern = op.join(drive_path, "image_02", "data", "*.png")
        frame_paths = glob(frame_pattern)
        frame_paths.sort()
        frame_paths = frame_paths[snippet_len // 2:-snippet_len // 2]
        frame_files = []
        # reformat file paths into "kitti_static_frame.txt" format
        for frame in frame_paths:
            splits = frame.strip("\n").split("/")
            # format: 'date drive_id frame_id' e.g. '2011_09_26 0001 0000000000'
            frame_files.append(f"{splits[-5]} {splits[-4][-9:-5]} {splits[-1][:-4]}")

        frame_files = self.remove_static_frames(frame_files)
        # convert to frame name to int
        frame_inds = [int(frame.split()[-1]) for frame in frame_files]
        frame_inds.sort()
        print("[frame_indices] frame ids:", frame_inds)
        return frame_inds


class KittiRawTestUtil(KittiUtil):
    def __init__(self):
        super().__init__()

    def create_drive_loader(self, base_path, drive):
        date, drive_id = drive
        return pykitti.raw(base_path, date, drive_id)

    def frame_indices(self, drive_path, snippet_len):
        print("drive path", drive_path)
        # count total frames in drive
        frame_pattern = op.join(drive_path, "image_02", "data", "*.png")
        num_frames = len(glob(frame_pattern))
        drive_splits = drive_path.split("/")
        # format drive_path into 'date drive'
        drive_id = f"{drive_splits[-2]} {drive_splits[-1][-9:-5]}"
        filename = op.join(op.dirname(op.abspath(__file__)), "resources", "kitti_test_depth_frames.txt")

        with open(filename, "r") as fr:
            lines = fr.readlines()
            test_frames = [line.strip("\n") for line in lines if line.startswith(drive_id)]
            test_frames = self.remove_static_frames(test_frames)
            frame_inds = [int(frame.split()[-1]) for frame in test_frames]
            frame_inds = [index for index in frame_inds if 2 <= index < num_frames-2]
            frame_inds.sort()
            print("test frames:", frame_inds)

        return frame_inds

