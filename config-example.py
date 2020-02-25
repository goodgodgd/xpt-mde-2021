import os.path as op


class VodeOptions:
    """
    data options
    """
    STEREO = True
    SNIPPET_LEN = 5
    IM_WIDTH = 384
    IM_HEIGHT = 128
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    """
    training options
    """
    BATCH_SIZE = 8
    EPOCHS = 51
    LEARNING_RATE = 0.0001
    ENABLE_SHAPE_DECOR = False
    CKPT_NAME = "vode2"
    LOG_LOSS = True

    """
    path options
    """
    DATAPATH = "/media/ian/IanPrivatePP/Datasets/vode_data_384_stereo_clip"
    assert(op.isdir(DATAPATH))
    DATAPATH_SRC = op.join(DATAPATH, "srcdata")
    DATAPATH_TFR = op.join(DATAPATH, "tfrecords")
    DATAPATH_CKP = op.join(DATAPATH, "checkpts")
    DATAPATH_LOG = op.join(DATAPATH, "log")
    DATAPATH_PRD = op.join(DATAPATH, "prediction")
    DATAPATH_EVL = op.join(DATAPATH, "evaluation")
    PROJECT_ROOT = op.dirname(__file__)

    """
    model options: network architecture, loss wegihts, ...
    """
    STEREO_EXTRINSIC = True
    SSIM_RATIO = 0.8
    LOSS_WEIGHTS = {"L1": (1. - SSIM_RATIO)*1., "SSIM": SSIM_RATIO*0.5, "smoothe": 1.,
                    "L1_R": (1. - SSIM_RATIO)*1., "SSIM_R": SSIM_RATIO*0.5, "smoothe_R": 1.,
                    "stereo_L1": 0.04, "stereo_pose": 0.5}
    DATASET = "kitti_raw"
    NET_NAMES = {"depth": "NASNetMobile", "camera": "PoseNet"}
    SYNTHESIZER = "SynthesizeMultiScale"
    OPTIMIZER = "adam_constant"
    PRETRAINED_WEIGHT = True


opts = VodeOptions()


# TODO: add or change RAW_DATA_PATHS as dataset paths in your PC
RAW_DATA_PATHS = {
    "kitti_raw": "/media/ian/IanPrivatePP/Datasets/kitti_raw_data",
    "kitti_odom": "/media/ian/IanPrivatePP/Datasets/kitti_odometry",
}


class WrongDatasetException(Exception):
    def __init__(self, msg):
        super().__init__(msg)


def get_raw_data_path(dataset_name):
    if dataset_name in RAW_DATA_PATHS:
        dataset_path = RAW_DATA_PATHS[dataset_name]
        assert op.isdir(dataset_path)
        return dataset_path
    else:
        raise WrongDatasetException(f"Unavailable dataset name, available datasets are {list(RAW_DATA_PATHS.keys())}")


import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=100)
