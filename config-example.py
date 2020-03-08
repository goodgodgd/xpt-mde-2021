import os.path as op

# TODO: add or change RAW_DATA_PATHS as dataset paths in your PC
RAW_DATA_PATHS = {
    "kitti_raw": "/media/ian/IanPrivatePP/Datasets/kitti_raw_data",
    "kitti_odom": "/media/ian/IanPrivatePP/Datasets/kitti_odometry",
    "cityscapes": "/media/ian/IanPrivatePP/Datasets/cityscapes",
    "cityscapes_seq": "/media/ian/IanPrivatePP/Datasets/cityscapes",
}
RESULT_DATAPATH = "/media/ian/IanPrivatePP/Datasets/vode_data_384"


class VodeOptions:
    """
    data options
    """
    STEREO = False
    SNIPPET_LEN = 5
    IM_WIDTH = 384
    IM_HEIGHT = 128
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    VALIDATION_FRAMES = 500
    DATASETS_TO_PREPARE = {"kitti_raw": ["train", "test", "val"],
                           "kitti_odom": ["train", "test", "val"],
                           "cityscapes": ["train_extra", "test"],
                           "cityscapes_seq": ["train", "test", "val"],
                           }

    """
    training options
    """
    BATCH_SIZE = 4
    EPOCHS = 51
    LEARNING_RATE = 0.0001
    ENABLE_SHAPE_DECOR = False
    CKPT_NAME = "vode1_exp_acti"
    LOG_LOSS = True

    """
    path options
    """
    DATAPATH = RESULT_DATAPATH
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
    DATASET_TO_USE = "kitti_raw"
    STEREO_EXTRINSIC = True
    SSIM_RATIO = 0.8
    LOSS_WEIGHTS = {"L1": (1. - SSIM_RATIO)*1., "SSIM": SSIM_RATIO*0.5, "smoothe": 1.,
                    "L1_R": (1. - SSIM_RATIO)*1., "SSIM_R": SSIM_RATIO*0.5, "smoothe_R": 1.,
                    "stereo_L1": 0.04, "stereo_pose": 0.5}
    NET_NAMES = {"depth": "NASNetMobile", "camera": "PoseNet"}
    SYNTHESIZER = "SynthesizeMultiScale"
    OPTIMIZER = "adam_constant"
    DEPTH_ACTIVATION = "InverseSigmoid"
    PRETRAINED_WEIGHT = True


opts = VodeOptions()


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
