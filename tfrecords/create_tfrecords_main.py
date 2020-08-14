import os.path as op
import numpy as np

import settings
from config import opts
from utils.util_class import WrongInputException
import tfrecords.tfrecord_maker as tm


def convert_to_tfrecords_directly():
    datasets = opts.DATASETS_TO_PREPARE
    for dataset, splits in datasets.items():
        for split in splits:
            tfrpath = op.join(opts.DATAPATH_TFR, f"{dataset.split('__')[0]}_{split}")
            if op.isdir(tfrpath):
                print("[convert_to_tfrecords] tfrecord already created in", tfrpath)
                continue

            srcpath = opts.get_raw_data_path(dataset)
            tfrmaker = tfrecord_maker_factory(dataset, split, srcpath, tfrpath)
            tfrmaker.make(opts.DRIVE_LIMIT, opts.FRAME_LIMIT)


def tfrecord_maker_factory(dataset, split, srcpath, tfrpath):
    dstshape = opts.get_shape("SHWC", dataset.split('__')[0])
    if dataset == "kitti_raw":
        return tm.KittiRawTfrecordMaker(dataset, split, srcpath, tfrpath, 2000, opts.STEREO, dstshape)
    elif dataset == "kitti_odom":
        return tm.KittiOdomTfrecordMaker(dataset, split, srcpath, tfrpath, 2000, opts.STEREO, dstshape)
    elif dataset.startswith("cityscapes"):
        return tm.CityscapesTfrecordMaker(dataset, split, srcpath, tfrpath, 2000, opts.STEREO, dstshape)
    elif dataset is "waymo":
        return tm.WaymoTfrecordMaker(dataset, split, srcpath, tfrpath, 2000, opts.STEREO, dstshape)
    elif dataset is "driving_stereo":
        return tm.DrivingStereoTfrecordMaker(dataset, split, srcpath, tfrpath, 2000, opts.STEREO, dstshape)
    else:
        WrongInputException(f"Invalid dataset: {dataset}")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    # convert_to_tfrecords()
    convert_to_tfrecords_directly()
