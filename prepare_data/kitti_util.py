import os.path as op
from glob import glob
import pykitti


class KittiUtil:
    def __init__(self):
        self.static_frames = self.read_static_frames("resources/kitti_static_frames.txt")

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
        raise NotImplementedError()

    def create_drive_loader(self, base_path, drive):
        raise NotImplementedError()

    def frame_indices(self, num_frames, snippet_len):
        raise NotImplementedError()


class KittiRawTrainUtil(KittiUtil):
    def list_drives(self, dataset, split):
        filename = f"resources/{dataset}_{split}_scenes.txt"
        with open(filename, "r") as f:
            drives = f.readlines()
            drives = [tuple(drive.strip("\n").split()) for drive in drives]
            return drives

    def create_drive_loader(self, base_path, drive):
        date, drive_id = drive
        return pykitti.raw(base_path, date, drive_id)

    def frame_indices(self, drive_path, snippet_len):
        frame_pattern = op.join(drive_path, "image_02", "data", "*.png")
        frame_paths = glob(frame_pattern)
        frame_files = []
        for frame in frame_paths:
            splits = frame.strip("\n").split("/")
            # format: 'date drive_id frame_id' e.g. '2011_09_26 0001 0000000000'
            frame_files.append(f"{splits[-5]} {splits[-4][-9:-5]} {splits[-1][:-4]}")
        frame_files = self.remove_static_frames(frame_files)
        frame_ids = [int(frame.split()[-1]) for frame in frame_files]
        frame_ids.sort()
        print("[frame_indices] frame ids:", frame_ids)
        return frame_ids[snippet_len//2:-snippet_len//2]


class KittiRawTestUtil(KittiUtil):
    def list_drives(self, dataset, split):
        filename = f"resources/{dataset}_{split}_scenes.txt"
        with open(filename, "r") as f:
            drives = f.readlines()
            drives = [tuple(drive.strip("\n").split()) for drive in drives]
            return drives

    def create_drive_loader(self, base_path, drive):
        date, drive_id = drive
        return pykitti.raw(base_path, date, drive_id)

    def frame_indices(self, num_frames, snippet_len):
        return range(snippet_len//2, num_frames - snippet_len//2)
