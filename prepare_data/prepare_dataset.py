from config import opts
from kitti_loader import KittiDataLoader


def prepare_input_data():
    for dataset in ["kitti_raw", "kitti_odom"]:
        for split in ["train", "test"]:
            loader = KittiDataLoader(opts.RAW_DATASET_PATH, dataset, split)
            prepare_and_save_snippets(loader)


def prepare_and_save_snippets(loader):
    for drive in loader.drives:
        loader.load_drive(drive)
        for snippet in loader.snippet_generator(opts.SNIPPET_LEN):
            frames = snippet["frames"]
            poses = snippet["gt_pose"]
            depths = snippet["gt_depth"]
            intrinsic = snippet["intrinsic"]
            # save them in options.snippet_path["kitti_raw"]
