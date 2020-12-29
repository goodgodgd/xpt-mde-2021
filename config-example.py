import os.path as op

# TODO: add or change RAW_DATA_PATHS as dataset paths in your PC
RAW_DATA_PATHS = {
    "kitti_raw": "/media/ian/IanBook2/datasets/kitti_raw_data",
    "kitti_odom": "/media/ian/IanBook2/datasets/kitti_odometry",
    "cityscapes__sequence": "/media/ian/IanBook2/datasets/raw_zips/cityscapes",
    "waymo": "/media/ian/IanBook2/datasets/waymo",
    "a2d2": "/media/ian/IanBook2/datasets/raw_zips/a2d2/zips",
}
RESULT_DATAPATH = "/media/ian/IanBook2/vode_data/vode_stereo_1227"


class FixedOptions:
    """
    data options
    """
    STEREO = True
    SNIPPET_LEN = 5
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    IMAGE_SIZES = {"kitti_raw": (128, 512),
                   "kitti_odom": (128, 512),
                   "cityscapes": (192, 448),
                   "waymo": (256, 384),
                   "a2d2": (256, 512),
                   "driving_stereo": (192, 384),
                   }

    """
    training options
    """
    PER_REPLICA_BATCH = 2
    BATCH_SIZE = PER_REPLICA_BATCH
    OPTIMIZER = ["adam_constant"][0]
    DEPTH_ACTIVATION = ["InverseSigmoid", "Exponential"][0]
    PRETRAINED_WEIGHT = True

    """
    network options: network architecture, convolution args, ... 
    """
    JOINT_NET = {"depth": ["DepthNetBasic", "DepthNetNoResize", "MobileNetV2", "NASNetMobile",
                           "DenseNet121", "VGG16", "Xception", "ResNet50V2", "NASNetLarge"][3],
                 "camera": "PoseNet",
                 "flow": "PWCNet"
                 }
    RIGID_NET = {"depth": JOINT_NET["depth"], "camera": JOINT_NET["camera"]}
    FLOW_NET = {"flow": JOINT_NET["flow"]}
    DEPTH_CONV_ARGS = {"activation": "leaky_relu", "activation_param": 0.1,
                       "kernel_initializer": "truncated_normal", "kernel_initializer_param": 0.025}
    DEPTH_UPSAMPLE_INTERP = "nearest"
    POSE_CONV_ARGS = {"activation": "leaky_relu", "activation_param": 0.1,
                      "kernel_initializer": "truncated_normal", "kernel_initializer_param": 0.025}
    FLOW_CONV_ARGS = {"activation": "leaky_relu", "activation_param": 0.1,
                      "kernel_initializer": "truncated_normal", "kernel_initializer_param": 0.025}


class VodeOptions(FixedOptions):
    """
    path options
    """
    CKPT_NAME = "vode7"

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
    data options
    """
    DATASETS_TO_PREPARE = {"kitti_raw": ["train", "test"],
                           "kitti_odom": ["train", "test"],
                           "a2d2": ["train"],
                           "cityscapes__sequence": ["train"],
                           "waymo": ["train"],
                           }
    # only when making small tfrecords to test training
    FRAME_PER_DRIVE = 0
    TOTAL_FRAME_LIMIT = 0
    VALIDATION_FRAMES = 500
    AUGMENT_PROBS = {"CropAndResize": 0.2,
                     "HorizontalFlip": 0.2,
                     "ColorJitter": 0.2}

    """
    training options
    """
    ENABLE_SHAPE_DECOR = False
    LOG_LOSS = True
    TRAIN_MODE = ["eager", "graph", "distributed"][1]
    SSIM_RATIO = 0.8
    LOSS_RIGID_T1 = {
        "L1": (1. - SSIM_RATIO) * 1., "L1_R": (1. - SSIM_RATIO) * 1.,
        "SSIM": SSIM_RATIO * 0.5, "SSIM_R": SSIM_RATIO * 0.5,
        "smoothe": 1., "smoothe_R": 1.,
        "stereoL1": 0.01, "stereoSSIM": 0.01,
        "stereoPose": 1.,
    }
    LOSS_RIGID_T2 = {
        "L1": (1. - SSIM_RATIO) * 1., "L1_R": (1. - SSIM_RATIO) * 1.,
        "SSIM": SSIM_RATIO * 0.5, "SSIM_R": SSIM_RATIO * 0.5,
        "smoothe": 1., "smoothe_R": 1.,
        "stereoL1": (1. - SSIM_RATIO), "stereoSSIM": SSIM_RATIO,
        "stereoPose": 1.,
    }
    LOSS_RIGID_T3 = {
        "md2L1": (1. - SSIM_RATIO) * 1., "md2L1_R": (1. - SSIM_RATIO) * 1.,
        "md2SSIM": SSIM_RATIO * 0.5, "md2SSIM_R": SSIM_RATIO * 0.5,
        "cmbL1": (1. - SSIM_RATIO) * 1., "cmbL1_R": (1. - SSIM_RATIO) * 1.,
        "cmbSSIM": SSIM_RATIO * 0.5, "cmbSSIM_R": SSIM_RATIO * 0.5,
        "smoothe": 1., "smoothe_R": 1.,
        "stereoL1": 0.01, "stereoSSIM": 0.01,
        "stereoPose": 1.,
    }
    LOSS_FLOW = {
        "flowL2": 1., "flowL2_R": 1.,
        "flow_reg": 4e-7
    }

    PRE_TRAINING_PLAN = [
        # pretraining rigid net
        (FixedOptions.RIGID_NET, "kitti_raw",   5, 0.0001, LOSS_RIGID_T1, True),
        (FixedOptions.RIGID_NET, "kitti_raw",   5, 0.0001, LOSS_RIGID_T2, True),
        (FixedOptions.RIGID_NET, "kitti_odom",  5, 0.0001, LOSS_RIGID_T1, True),
        (FixedOptions.RIGID_NET, "kitti_odom",  5, 0.0001, LOSS_RIGID_T2, True),
    ]

    FINE_TRAINING_PLAN_KITTI_RAW = [
        (FixedOptions.FLOW_NET, "kitti_raw",    10, 0.00001, LOSS_FLOW, True),
        (FixedOptions.RIGID_NET, "kitti_raw",    5, 0.0001, LOSS_RIGID_T1, True),
        (FixedOptions.JOINT_NET, "kitti_raw",   10, 0.0001, LOSS_RIGID_T2, True),
        (FixedOptions.JOINT_NET, "kitti_raw",   10, 0.00001, LOSS_RIGID_T2, True),
    ]

    FINE_TRAINING_PLAN_KITTI_ODOM = [
        (FixedOptions.FLOW_NET, "kitti_odom",   10, 0.00001, LOSS_FLOW, True),
        (FixedOptions.RIGID_NET, "kitti_odom",   5, 0.0001, LOSS_RIGID_T1, True),
        (FixedOptions.JOINT_NET, "kitti_odom",  10, 0.0001, LOSS_RIGID_T2, True),
        (FixedOptions.JOINT_NET, "kitti_odom",  10, 0.00001, LOSS_RIGID_T2, True),
    ]

    FINE_TRAINING_PLAN = FINE_TRAINING_PLAN_KITTI_RAW
    TEST_PLAN = [
        (FixedOptions.RIGID_NET, "kitti_raw",  ["depth"], "ep15"),
        (FixedOptions.RIGID_NET, "kitti_odom", ["pose"],  "ep15"),
    ]

    @classmethod
    def get_raw_data_path(cls, dataset_name):
        if dataset_name in RAW_DATA_PATHS:
            dataset_path = RAW_DATA_PATHS[dataset_name]
            assert op.exists(dataset_path), f"{dataset_path}"
            return dataset_path
        else:
            assert 0, f"Invalid dataset name, available datasets are {list(RAW_DATA_PATHS.keys())}"

    @classmethod
    def get_img_shape(cls, code="HW", dataset="kitti_raw", scale_div=1):
        imsize = cls.IMAGE_SIZES[dataset]
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
print(f"[config] ckpt path: {opts.DATAPATH_CKP}, nets: {opts.JOINT_NET}")

import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=150)
