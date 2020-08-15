import os.path as op

# TODO: add or change RAW_DATA_PATHS as dataset paths in your PC
RAW_DATA_PATHS = {
    "kitti_raw": "/media/ian/IanBook/datasets/kitti_raw_data",
    "kitti_odom": "/media/ian/IanBook/datasets/kitti_odometry",
    "cityscapes__extra": "/media/ian/IanBook/datasets/raw_zips/cityscapes",
    "cityscapes__sequence": "/media/ian/IanBook/datasets/raw_zips/cityscapes",
    "waymo": "/media/ian/IanBook/datasets/waymo",
    "driving_stereo": "/media/ian/IanBook/datasets/raw_zips/driving_stereo",
}
# RESULT_DATAPATH = "/home/ian/workspace/vode/vode-data"
RESULT_DATAPATH = "/media/ian/IanBook/vode_data/vode_stereo_0801"

IMAGE_SIZES = {
    "kitti_raw": (128, 384),
    "kitti_odom": (128, 384),
    "cityscapes": (196, 384),
    "waymo": (256, 384),
    "driving_stereo": (196, 384),
}


class VodeOptions:
    """
    data options
    """
    STEREO = True
    SNIPPET_LEN = 5
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    # cityscapes__sequence MUST be after cityscapes__extra
    DATASETS_TO_PREPARE = {"kitti_raw": ["train", "test"],
                           "kitti_odom": ["train", "test"],
                           "cityscapes__extra": ["train"],
                           "cityscapes__sequence": ["train"],
                           "waymo": ["train"],
                           "driving_stereo": ["train", "test"],
                           }
    # only when making small tfrecords to test training
    FRAME_PER_DRIVE = 100
    TOTAL_FRAME_LIMIT = 500
    VALIDATION_FRAMES = 300
    AUGMENT_PROBS = {"CropAndResize": 0.2,
                     "HorizontalFlip": 0.2,
                     "ColorJitter": 0.2}

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
    CKPT_NAME = "vode7"
    PER_REPLICA_BATCH = 4
    BATCH_SIZE = PER_REPLICA_BATCH
    EPOCHS = 51
    LEARNING_RATE = 0.0001
    ENABLE_SHAPE_DECOR = False
    LOG_LOSS = True
    TRAIN_MODE = ["eager", "graph", "distributed"][1]
    DATASET_TO_USE = ["kitti_raw", "kitti_odom"][0]
    STEREO_EXTRINSIC = True
    SSIM_RATIO = 0.8
    LOSS_WEIGHTS = {"L1": (1. - SSIM_RATIO) * 1., "L1_R": (1. - SSIM_RATIO) * 1.,
                    "SSIM": SSIM_RATIO * 0.5, "SSIM_R": SSIM_RATIO * 0.5,
                    "smoothe": 1., "smoothe_R": 1.,
                    "stereo_L1": 0.01, "stereo_SSIM": 0.01,
                    "stereo_pose": 1.,
                    "flow_L2": 1., "flow_L2_R": 1.,
                    "flow_L2_reg": 4e-7
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

    @classmethod
    def get_raw_data_path(cls, dataset_name):
        if dataset_name in RAW_DATA_PATHS:
            dataset_path = RAW_DATA_PATHS[dataset_name]
            assert op.exists(dataset_path), f"{dataset_path}"
            return dataset_path
        else:
            assert 0, f"Invalid dataset name, available datasets are {list(RAW_DATA_PATHS.keys())}"

    @classmethod
    def get_shape(cls, code="HW", dataset=DATASET_TO_USE, scale_div=1):
        imsize = IMAGE_SIZES[dataset]
        if code is "H":
            return imsize[0] // scale_div
        elif code is "W":
            return imsize[1] // scale_div
        elif code is "HW":
            return imsize
        elif code is "WH":
            return imsize[1] // scale_div, imsize[0] // scale_div
        elif code is "HWC":
            return imsize[0] // scale_div, imsize[1] // scale_div, 3
        elif code is "SHW":
            return cls.SNIPPET_LEN, imsize[0] // scale_div, imsize[1] // scale_div
        elif code is "SHWC":
            return cls.SNIPPET_LEN, imsize[0] // scale_div, imsize[1] // scale_div, 3
        elif code is "BSHWC":
            return cls.BATCH_SIZE, cls.SNIPPET_LEN, imsize[0] // scale_div, imsize[1] // scale_div, 3
        elif code is "RSHWC":
            return cls.PER_REPLICA_BATCH, cls.SNIPPET_LEN, imsize[0] // scale_div, imsize[1] // scale_div, 3
        else:
            assert 0, f"Invalid code: {code}"


opts = VodeOptions()


import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=100)
