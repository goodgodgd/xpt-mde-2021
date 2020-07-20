import os.path as op

# TODO: add or change RAW_DATA_PATHS as dataset paths in your PC
RAW_DATA_PATHS = {
    "kitti_raw": "/media/ian/IanPrivatePP/Datasets/kitti_raw_data",
    "kitti_odom": "/media/ian/IanPrivatePP/Datasets/kitti_odometry",
    "cityscapes": "/media/ian/IanPrivatePP/Datasets/cityscapes",
    "cityscapes_seq": "/media/ian/IanPrivatePP/Datasets/cityscapes",
}
RESULT_DATAPATH = "/media/ian/IanPrivatePP/Datasets/vode_data/vode_stereo_0705"


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
    VALIDATION_FRAMES = 500
    DATASETS_TO_PREPARE = {"kitti_raw": ["train", "test", "val"],
                           # "kitti_odom": ["train", "test", "val"],
                           # "cityscapes": ["train_extra"],
                           # "cityscapes_seq": ["train"],
                           }
    # only when making small tfrecords to test training
    LIMIT_FRAMES = None
    SHUFFLE_TFRECORD_INPUT = False

    """
    path options
    """
    DATAPATH = RESULT_DATAPATH
    assert(op.isdir(DATAPATH))
    DATAPATH_SRC = op.join(DATAPATH, "srcdata")
    DATAPATH_TFR = op.join(DATAPATH, "tfrecords_small")
    DATAPATH_CKP = op.join(DATAPATH, "checkpts")
    DATAPATH_LOG = op.join(DATAPATH, "log")
    DATAPATH_PRD = op.join(DATAPATH, "prediction")
    DATAPATH_EVL = op.join(DATAPATH, "evaluation")
    PROJECT_ROOT = op.dirname(__file__)

    """
    training options
    """
    CKPT_NAME = "vode2"
    PER_REPLICA_BATCH = 4
    BATCH_SIZE = PER_REPLICA_BATCH
    EPOCHS = 51
    LEARNING_RATE = 0.0001
    ENABLE_SHAPE_DECOR = False
    LOG_LOSS = True
    TRAIN_MODE = ["eager", "graph", "distributed"][2]
    DATASET_TO_USE = ["kitti_raw", "kitti_odom"][0]
    STEREO_EXTRINSIC = True
    SSIM_RATIO = 0.8
    LOSS_WEIGHTS = {"L1": (1. - SSIM_RATIO) * 1., "SSIM": SSIM_RATIO * 0.5, "smoothe": 1.,
                    "L1_R": (1. - SSIM_RATIO) * 1., "SSIM_R": SSIM_RATIO * 0.5, "smoothe_R": 1.,
                    "stereo_L1": 0.01, "stereo_pose": 0.5,
                    "FW_L2": 1.,
                    "FW_L2_R": 1.,
                    "FW_L2_regular": 0.0004
                    }
    OPTIMIZER = ["adam_constant"][0]
    DEPTH_ACTIVATION = ["InverseSigmoid", "Exponential"][0]
    PRETRAINED_WEIGHT = True

    """
    network options: network architecture, convolution args, ... 
    """
    NET_NAMES = {"depth": "NASNetMobile",
                 "camera": "PoseNet",
                 "flow": "PWCNet"
                 }
    DEPTH_CONV_ARGS = {"activation": "leaky_relu", "activation_param": 0.1,
                       "kernel_initializer": "truncated_normal", "kernel_initializer_param": 0.025}
    DEPTH_UPSAMPLE_INTERP = "nearest"
    POSE_CONV_ARGS = {"activation": "leaky_relu", "activation_param": 0.1,
                      "kernel_initializer": "truncated_normal", "kernel_initializer_param": 0.025}
    FLOW_CONV_ARGS = {"activation": "leaky_relu", "activation_param": 0.1,
                      "kernel_initializer": "truncated_normal", "kernel_initializer_param": 0.025}


opts = VodeOptions()


def get_raw_data_path(dataset_name):
    if dataset_name in RAW_DATA_PATHS:
        dataset_path = RAW_DATA_PATHS[dataset_name]
        assert op.isdir(dataset_path)
        return dataset_path
    else:
        assert 0, f"Unavailable dataset name, available datasets are {list(RAW_DATA_PATHS.keys())}"


import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=100)
