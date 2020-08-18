import os.path as op
import numpy as np

import settings
from config import opts
import tfrecords.tfrecord_maker as tm
from tfrecords.validation_maker import generate_validation_tfrecords


def convert_to_tfrecords_directly():
    datasets = opts.DATASETS_TO_PREPARE
    for dataset, splits in datasets.items():
        for split in splits:
            tfrpath = op.join(opts.DATAPATH_TFR, f"{dataset.split('__')[0]}_{split}")
            if op.isdir(tfrpath):
                print("[convert_to_tfrecords] tfrecord already created in", op.basename(tfrpath))
                continue

            srcpath = opts.get_raw_data_path(dataset)
            tfrmaker = tfrecord_maker_factory(dataset, split, srcpath, tfrpath)
            tfrmaker.make(opts.FRAME_PER_DRIVE, opts.TOTAL_FRAME_LIMIT)

        # create validation split from test or train dataset
        tfrpath = op.join(opts.DATAPATH_TFR, f"{dataset.split('__')[0]}_val")
        if op.isdir(tfrpath):
            print("[convert_to_tfrecords] tfrecord already created in", op.basename(tfrpath))
            continue
        generate_validation_tfrecords(tfrpath, opts.VALIDATION_FRAMES)


def tfrecord_maker_factory(dataset, split, srcpath, tfrpath):
    dstshape = opts.get_img_shape("SHWC", dataset.split('__')[0])
    crop = False if dataset not in opts.IMAGE_CROP else opts.IMAGE_CROP[dataset]
    if dataset == "kitti_raw":
        return tm.KittiRawTfrecordMaker(dataset, split, srcpath, tfrpath, 2000, opts.STEREO, dstshape, crop)
    elif dataset == "kitti_odom":
        return tm.KittiOdomTfrecordMaker(dataset, split, srcpath, tfrpath, 2000, opts.STEREO, dstshape, crop)
    elif dataset.startswith("cityscapes"):
        return tm.CityscapesTfrecordMaker(dataset, split, srcpath, tfrpath, 2000, opts.STEREO, dstshape)
    elif dataset is "waymo":
        return tm.WaymoTfrecordMaker(dataset, split, srcpath, tfrpath, 2000, opts.STEREO, dstshape)
    elif dataset is "driving_stereo":
        return tm.DrivingStereoTfrecordMaker(dataset, split, srcpath, tfrpath, 2000, opts.STEREO, dstshape)
    else:
        assert 0, f"Invalid dataset: {dataset}"


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    convert_to_tfrecords_directly()
