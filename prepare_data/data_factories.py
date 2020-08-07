from config import opts
from utils.util_class import WrongInputException
import prepare_data.readers.kitti_reader as kr
import prepare_data.readers.city_reader as cr
import prepare_data.example_maker as em
import prepare_data.list_drives as ld

"""
ExampleMaker: generates training example, main function is snippet_generator()
DataLister: list drives (image sequence folders)
DatasetReader: reads data from files
"""


def example_maker_factory(raw_data_path, pose_avail, depth_avail, stereo=opts.STEREO, snippet_len=opts.SNIPPET_LEN):
    if stereo:
        snippet_maker = em.ExampleMakerStereo(raw_data_path, pose_avail, depth_avail, snippet_len)
    else:
        snippet_maker = em.ExampleMaker(raw_data_path, pose_avail, depth_avail, snippet_len)
    return snippet_maker


def drive_lister_factory(dataset, split, raw_data_path, save_path):
    if dataset == "kitti_raw":
        drive_lister = ld.KittiRawListDrives(raw_data_path, save_path, split)
    elif dataset == "kitti_odom":
        drive_lister = ld.KittiOdomListDrives(raw_data_path, save_path, split)
    elif dataset == "cityscapes":
        drive_lister = ld.CityListDrives(raw_data_path, save_path, split)
    elif dataset == "cityscapes_seq":
        drive_lister = ld.CityListDrives(raw_data_path, save_path, split, "_sequence")
    else:
        raise WrongInputException(f"No dataset and split like: {dataset}, {split}")
    return drive_lister


def dataset_reader_factory(raw_data_path, drive_path, dataset, split, stereo=opts.STEREO):
    if dataset == "kitti_raw" and split == "train":
        data_reader = kr.KittiRawTrainReader(raw_data_path, drive_path, stereo)
    elif dataset == "kitti_raw" and split == "test":
        data_reader = kr.KittiRawTestReader(raw_data_path, drive_path, stereo)
    elif dataset == "kitti_odom" and split == "train":
        data_reader = kr.KittiOdomTrainReader(raw_data_path, drive_path, stereo)
    elif dataset == "kitti_odom" and split == "test":
        data_reader = kr.KittiOdomTestReader(raw_data_path, drive_path, stereo)
    elif dataset == "cityscapes":
        data_reader = cr.CityScapesReader(raw_data_path, drive_path, stereo, split)
    elif dataset == "cityscapes_seq":
        data_reader = cr.CityScapesReader(raw_data_path, drive_path, stereo, split, "_sequence")
    else:
        raise WrongInputException(f"Wrong dataset and split: {dataset}, {split}")
    return data_reader

