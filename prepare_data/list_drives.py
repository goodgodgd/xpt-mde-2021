import os
import os.path as op
from glob import glob
from utils.util_class import WrongInputException


class ListDrivesBase:
    def __init__(self, srcpath, dstpath, split):
        self.srcpath = srcpath
        self.dstpath = dstpath
        self.split = split
        self.pose_avail = False
        self.depth_avail = False

    def list_drive_paths(self):
        """
        :return: directory paths to sequences of images
        """
        raise NotImplementedError()

    def make_saving_paths(self, drive_path):
        """
        :param drive_path: path to source sequence data
        :return: [image_path, pose_path, depth_path]
                 specific paths under "dstpath" to save image, pose, and depth
        """
        raise NotImplementedError()


class KittiRawListDrives(ListDrivesBase):
    def __init__(self, srcpath, dstpath, split):
        super().__init__(srcpath, dstpath, split)
        self.pose_avail = True
        self.depth_avail = True

    def list_drive_paths(self):
        prepare_data_path = op.dirname(op.abspath(__file__))
        filename = op.join(prepare_data_path, "resources", f"kitti_raw_{self.split}_scenes.txt")
        with open(filename, "r") as f:
            drives = f.readlines()
            drives.sort()
            drives = [tuple(drive.strip("\n").split()) for drive in drives]
            drives = [self._make_raw_data_path(drive) for drive in drives]
            print("[list_drive_paths] drive list:", [op.basename(drive) for drive in drives])

        return drives

    def _make_raw_data_path(self, drive):
        drive_path = op.join(self.srcpath, drive[0], f"{drive[0]}_drive_{drive[1]}_sync")
        return drive_path

    def make_saving_paths(self, drive_path):
        date, drive_id = self._parse_drive_path(drive_path)
        image_path = op.join(self.dstpath, f"{date}_{drive_id}")
        pose_path = op.join(image_path, "pose") if self.pose_avail else None
        depth_path = op.join(image_path, "depth") if self.depth_avail else None
        return image_path, pose_path, depth_path

    def _parse_drive_path(self, drive_path):
        dirsplits = op.basename(drive_path).split("_")
        date = f"{dirsplits[0]}_{dirsplits[1]}_{dirsplits[2]}"
        drive_id = dirsplits[4]
        return date, drive_id


class KittiOdomListDrives(ListDrivesBase):
    def __init__(self, srcpath, dstpath, split):
        super().__init__(srcpath, dstpath, split)
        self.pose_avail = True if split is "test" else False

    def list_drive_paths(self):
        if self.split is "train":
            drives = [f"{i:02d}" for i in range(11, 22)]
        else:
            drives = [f"{i:02d}" for i in range(0, 11)]
        drives = [self._make_raw_data_path(drive) for drive in drives]
        print("[list_drive_paths] drive list:", [op.basename(drive) for drive in drives])
        return drives

    def _make_raw_data_path(self, drive):
        drive_path = op.join(self.srcpath, "sequences", drive)
        return drive_path

    def make_saving_paths(self, drive_path):
        drive = op.basename(drive_path)
        image_path = op.join(self.dstpath, drive)
        pose_path = op.join(image_path, "pose") if self.pose_avail else None
        depth_path = op.join(image_path, "depth") if self.depth_avail else None
        return image_path, pose_path, depth_path


class CityListDrives(ListDrivesBase):
    def __init__(self, srcpath, dstpath, split, dir_suffix=""):
        super().__init__(srcpath, dstpath, split)
        self.depth_avail = True
        self.left_img_dir = "leftImg8bit"
        self.dir_suffix = dir_suffix

    """
    Public methods used outside this class
    """

    def list_drive_paths(self):
        split_path = op.join(self.srcpath, self.left_img_dir + self.dir_suffix, self.split)
        if not op.isdir(split_path):
            raise WrongInputException("[list_sequence_paths] path does NOT exist:" + split_path)

        city_names = os.listdir(split_path)
        city_names = [city for city in city_names if op.isdir(op.join(split_path, city))]
        total_sequences = []
        for city in city_names:
            pattern = op.join(split_path, city, "*.png")
            files = glob(pattern)
            seq_numbers = [file.split("_")[-3] for file in files]
            seq_numbers = list(set(seq_numbers))
            seq_numbers.sort()
            seq_paths = [op.join(self.split, city, f"{city}_{seq}") for seq in seq_numbers]
            total_sequences.extend(seq_paths)

        # total_sequences: list of ["city/city_seqind"]
        print("[list_drive_paths]", total_sequences)
        return total_sequences

    def make_saving_paths(self, drive_path):
        """
        :param drive_path: sequence path like "train/bochum/bochum_000000"
        :return: [image_path, pose_path, depth_path]
                 specific paths under "dstpath" to save image, pose, and depth
        """
        # e.g. image_path = "opts.DATAPATH/srcdata/cityscapes_train/bochum/bochum_000000"
        image_path = op.join(self.dstpath, op.basename(drive_path))
        pose_path = op.join(image_path, "pose") if self.pose_avail else None
        depth_path = op.join(image_path, "depth") if self.depth_avail else None
        return image_path, pose_path, depth_path



