import os.path as op
import numpy as np

# TODO: add or change RAW_DATA_PATHS as dataset paths in your PC
RAW_DATA_PATHS = {
    "kitti_raw": "/media/ri-bear/IanBook8T/datasets/kitti_raw_data",
    "kitti_odom": "/media/ri-bear/IntHDD/datasets/kitti_odometry",
    "cityscapes__sequence": "/media/ri-bear/IntHDD/datasets/raw_zips/cityscapes",
    "waymo": "/media/ri-bear/IntHDD/datasets/waymo",
    "a2d2": "/media/ri-bear/IntHDD/datasets/raw_zips/a2d2/zips",
}
RESULT_DATAPATH_LOW = "/media/ri-bear/IntHDD/vode_data/vode_0724_test"
RESULT_DATAPATH_HIGH = "/media/ri-bear/IntHDD/vode_data/vode_0121_2x"


class FixedOptions:
    """
    data options
    """
    STEREO = True
    HIGH_RES = True
    SNIPPET_LEN = 5
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    IMAGE_SIZES_SMALL = {"kitti_raw": (128, 512),  # 2:8 65536
                         "kitti_odom": (128, 512),
                         "cityscapes": (192, 512),  # 3:8
                         "waymo": (256, 384),  # 4:6 98304
                         "a2d2": (192, 384),  # 3:6 73728
                         }
    IMAGE_SIZES_LARGE = {"kitti_raw": (256, 1024),  # 2:8
                         "kitti_odom": (256, 1024),
                         "cityscapes": (384, 1024),  # 3:8
                         "waymo": (512, 768),  # 4:6
                         "a2d2": (384, 768),  # 3:6
                         }
    IMAGE_SIZES = IMAGE_SIZES_LARGE if HIGH_RES else IMAGE_SIZES_SMALL

    """
    training options
    """
    PER_REPLICA_BATCH = 1
    BATCH_SIZE = PER_REPLICA_BATCH
    OPTIMIZER = ["adam_constant"][0]
    DEPTH_ACTIVATION = ["InverseSigmoid", "Exponential"][0]
    PRETRAINED_WEIGHT = True

    """
    network options: network architecture, convolution args, ... 
    """
    JOINT_NET = {"depth": ["DepthNetBasic", "DepthNetNoResize", "MobileNetV2", "NASNetMobile",
                           "DenseNet121", "VGG16", "Xception", "ResNet50V2", "NASNetLarge",     # 4~
                           "EfficientNetB0", "EfficientNetB3", "EfficientNetB5", "EfficientNetB7"][11],  # 9~
                 "camera": "PoseNetImproved",
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

    IMAGE_GRADIENT_FACTOR = 4
    SMOOTHNESS_FACTOR = 20
    SSIM_RATIO = 0.5
    SCALE_WEIGHT_T1 = np.array([0.25, 0.25, 0.25, 0.25]) * 4.
    SCALE_WEIGHT_T2 = np.array([0.1, 0.2, 0.3, 0.4]) * 4.


class LossOptions(FixedOptions):
    F = FixedOptions
    LOSS_RIGID_T1 = {
        "L1": (1. - F.SSIM_RATIO), "L1_R": (1. - F.SSIM_RATIO),
        "SSIM": F.SSIM_RATIO, "SSIM_R": F.SSIM_RATIO,
        "smoothe": 1., "smoothe_R": 1.,
        "stereoL1": 0.01, "stereoSSIM": 0.01,
        "stereoPose": 1.,
    }
    LOSS_RIGID_T2 = {
        "L1": (1. - F.SSIM_RATIO), "L1_R": (1. - F.SSIM_RATIO),
        "SSIM": F.SSIM_RATIO, "SSIM_R": F.SSIM_RATIO,
        "smoothe": F.SMOOTHNESS_FACTOR, "smoothe_R": F.SMOOTHNESS_FACTOR,
        "stereoL1": (1. - F.SSIM_RATIO), "stereoSSIM": F.SSIM_RATIO,
        "stereoPose": 1.,
    }
    LOSS_RIGID_COMB = {
        "cmbL1": (1. - F.SSIM_RATIO) * 10, "cmbL1_R": (1. - F.SSIM_RATIO) * 10,
        "cmbSSIM": F.SSIM_RATIO, "cmbSSIM_R": F.SSIM_RATIO,
        "smoothe": F.SMOOTHNESS_FACTOR, "smoothe_R": F.SMOOTHNESS_FACTOR,
        "stereoL1": (1. - F.SSIM_RATIO), "stereoSSIM": F.SSIM_RATIO,
        "stereoPose": 1.,
    }
    LOSS_RIGID_MOA = {
        "moaL1": (1. - F.SSIM_RATIO) * 10, "moaL1_R": (1. - F.SSIM_RATIO) * 10,
        "moaSSIM": F.SSIM_RATIO, "moaSSIM_R": F.SSIM_RATIO,
        "smoothe": F.SMOOTHNESS_FACTOR, "smoothe_R": F.SMOOTHNESS_FACTOR,
        "stereoPose": 1.,
    }
    LOSS_RIGID_MOA_WST = {
        "moaL1": (1. - F.SSIM_RATIO) * 10, "moaL1_R": (1. - F.SSIM_RATIO) * 10,
        "moaSSIM": F.SSIM_RATIO, "moaSSIM_R": F.SSIM_RATIO,
        "smoothe": F.SMOOTHNESS_FACTOR, "smoothe_R": F.SMOOTHNESS_FACTOR,
        "stereoL1": (1. - F.SSIM_RATIO), "stereoSSIM": F.SSIM_RATIO,
        "stereoPose": 1.,
    }
    LOSS_FLOW = {
        "flowL2": 1., "flowL2_R": 1.,
        "flow_reg": 4e-7
    }
    LOSS_RIGID_MD2 = {
        "md2L1": (1. - F.SSIM_RATIO), "md2L1_R": (1. - F.SSIM_RATIO),
        "md2SSIM": F.SSIM_RATIO, "md2SSIM_R": F.SSIM_RATIO,
        "smoothe": 1., "smoothe_R": 1.,
        "stereoL1": (1. - F.SSIM_RATIO), "stereoSSIM": F.SSIM_RATIO,
        "stereoPose": 1.,
    }

    """
    TRAINING PLAN to make best performance
    """
    LOSS_PRETRAIN_STEP3 = [LOSS_RIGID_T2, LOSS_RIGID_MOA_WST][0]
    LOSS_FINETUNE_STEP3 = [LOSS_RIGID_T2, LOSS_RIGID_MOA_WST, LOSS_RIGID_COMB][2]
    FINE_TUNE_NET = [FixedOptions.RIGID_NET, FixedOptions.JOINT_NET][1]
    TRAINING_PLAN_28 = [
        # pretraining
        (FixedOptions.RIGID_NET, "kitti_raw", 5, 0.00001, LOSS_RIGID_T1, F.SCALE_WEIGHT_T1, True),
        (FixedOptions.RIGID_NET, "kitti_raw", 10, 0.0001, LOSS_PRETRAIN_STEP3, F.SCALE_WEIGHT_T1, True),
        (FixedOptions.RIGID_NET, "a2d2", 10, 0.0001, LOSS_PRETRAIN_STEP3, F.SCALE_WEIGHT_T1, True),
        (FixedOptions.RIGID_NET, "waymo", 10, 0.0001, LOSS_RIGID_T2, F.SCALE_WEIGHT_T1, True),
        (FixedOptions.RIGID_NET, "kitti_odom", 10, 0.0001, LOSS_PRETRAIN_STEP3, F.SCALE_WEIGHT_T1, True),
        (FixedOptions.RIGID_NET, "cityscapes", 10, 0.00001, LOSS_PRETRAIN_STEP3, F.SCALE_WEIGHT_T1, True),
        (FixedOptions.RIGID_NET, "kitti_raw", 5, 0.0001, LOSS_PRETRAIN_STEP3, F.SCALE_WEIGHT_T1, True),
        # fine tuning
        (FINE_TUNE_NET, "kitti_raw", 10, 0.0001, LOSS_FINETUNE_STEP3, F.SCALE_WEIGHT_T1, True),
        (FINE_TUNE_NET, "kitti_raw", 10, 0.00001, LOSS_FINETUNE_STEP3, F.SCALE_WEIGHT_T1, True),
        (FINE_TUNE_NET, "kitti_raw", 5, 0.000001, LOSS_FINETUNE_STEP3, F.SCALE_WEIGHT_T1, True),
    ]
    """
    TRAINING PLAN without waymo (ablation study)
    """
    LOSS_PRETRAIN_STEP4 = LOSS_RIGID_T2
    LOSS_FINETUNE_STEP4 = LOSS_RIGID_COMB
    FINE_TUNE_NET = FixedOptions.JOINT_NET
    TRAINING_PLAN_29 = [
        # pretraining rigid net
        (FixedOptions.RIGID_NET, "kitti_raw", 5, 0.00001, LOSS_RIGID_T1, F.SCALE_WEIGHT_T1, True),
        (FixedOptions.RIGID_NET, "kitti_raw", 10, 0.0001, LOSS_PRETRAIN_STEP4, F.SCALE_WEIGHT_T1, True),
        (FixedOptions.RIGID_NET, "a2d2", 10, 0.0001, LOSS_PRETRAIN_STEP4, F.SCALE_WEIGHT_T1, True),
        (FixedOptions.RIGID_NET, "kitti_odom", 10, 0.0001, LOSS_PRETRAIN_STEP4, F.SCALE_WEIGHT_T1, True),
        (FixedOptions.RIGID_NET, "cityscapes", 10, 0.00001, LOSS_PRETRAIN_STEP4, F.SCALE_WEIGHT_T1, True),
        (FixedOptions.RIGID_NET, "kitti_raw", 5, 0.0001, LOSS_PRETRAIN_STEP4, F.SCALE_WEIGHT_T1, True),
        # fine tuning
        (FINE_TUNE_NET, "kitti_raw", 10, 0.0001, LOSS_FINETUNE_STEP4, F.SCALE_WEIGHT_T1, True),
        (FINE_TUNE_NET, "kitti_raw", 10, 0.00001, LOSS_FINETUNE_STEP4, F.SCALE_WEIGHT_T1, True),
        (FINE_TUNE_NET, "kitti_raw", 5, 0.000001, LOSS_FINETUNE_STEP4, F.SCALE_WEIGHT_T1, True),
    ]
    """
    TRAINING PLAN to compare backbones without multi dataset pretraining (ablation study)
    """
    TRAINING_PLAN_30 = [
        # pretraining rigid net
        (FixedOptions.RIGID_NET, "kitti_raw", 5, 0.00001, LOSS_RIGID_T1, F.SCALE_WEIGHT_T1, True),
        (FixedOptions.RIGID_NET, "kitti_raw", 10, 0.0001, LOSS_RIGID_T2, F.SCALE_WEIGHT_T1, True),
        (FixedOptions.RIGID_NET, "kitti_raw", 5, 0.0001, LOSS_RIGID_T2, F.SCALE_WEIGHT_T1, True),
        # fine-tune rigid net aided by flow net
        (FixedOptions.JOINT_NET, "kitti_raw", 10, 0.0001, LOSS_RIGID_COMB, F.SCALE_WEIGHT_T1, True),
        (FixedOptions.JOINT_NET, "kitti_raw", 10, 0.00001, LOSS_RIGID_COMB, F.SCALE_WEIGHT_T1, True),
        (FixedOptions.JOINT_NET, "kitti_raw", 5, 0.000001, LOSS_RIGID_COMB, F.SCALE_WEIGHT_T1, True),
    ]


class VodeOptions(LossOptions):
    L = LossOptions
    """
    path options
    """
    CKPT_NAME = "mde01"
    DEVICE = "/GPU:0"

    DATAPATH = RESULT_DATAPATH_HIGH if FixedOptions.HIGH_RES else RESULT_DATAPATH_LOW
    print(f"=== DATAPATH: {op.isdir(DATAPATH)}, {DATAPATH}")
    assert op.isdir(DATAPATH), f"=== DATAPATH: {DATAPATH}"
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
    DATASETS_TO_PREPARE = {"cityscapes__sequence": ["train"],
                           "waymo": ["train"],
                           "a2d2": ["train"],
                           "kitti_raw": ["train", "test"],
                           "kitti_odom": ["train", "test"],
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
    TRAINING_PLAN = L.TRAINING_PLAN_28
    RIGID_EF0 = {"depth": "EfficientNetB0", "camera": "PoseNetImproved", "flow": "PWCNet"}
    RIGID_EF3 = {"depth": "EfficientNetB3", "camera": "PoseNetImproved", "flow": "PWCNet"}
    RIGID_EF5 = {"depth": "EfficientNetB5", "camera": "PoseNetImproved", "flow": "PWCNet"}
    RIGID_EF7 = {"depth": "EfficientNetB7", "camera": "PoseNetImproved", "flow": "PWCNet"}
    RIGID_MOBILE = {"depth": "MobileNetV2", "camera": "PoseNetImproved", "flow": "PWCNet"}
    RIGID_NASMOB = {"depth": "NASNetMobile", "camera": "PoseNetImproved", "flow": "PWCNet"}

    TEST_PLAN_LOW = [
        (RIGID_EF5, "kitti_raw", ["depth"], "vode28_static", "latest"),
        (RIGID_EF5, "kitti_raw", ["depth"], "vode28_static_comb", "latest"),
        (RIGID_EF5, "kitti_raw", ["depth"], "vode28_static_moa", "latest"),
        (RIGID_EF5, "kitti_raw", ["depth"], "vode29_nowaymo", "latest"),
        (RIGID_EF0, "kitti_raw", ["depth"], "vode30_ef0", "latest"),
        (RIGID_EF3, "kitti_raw", ["depth"], "vode30_ef3", "latest"),
        (RIGID_EF5, "kitti_raw", ["depth"], "vode30_ef5", "latest"),
        (RIGID_EF7, "kitti_raw", ["depth"], "vode30_ef7", "latest"),
        (RIGID_MOBILE, "kitti_raw", ["depth"], "vode30_mobile", "latest"),
        (RIGID_NASMOB, "kitti_raw", ["depth"], "vode30_nasmob", "latest"),
    ]

    TEST_PLAN_HIGH = [
        (RIGID_EF5, "kitti_raw", ["depth"], "vode24_ef5_main_flownet", "latest"),
        (RIGID_EF5, "kitti_raw", ["depth"], "vode24_ef5_nowaymo", "latest"),
        (RIGID_EF5, "kitti_raw", ["depth"], "vode29_fullpt", "latest"),
        (RIGID_EF5, "kitti_raw", ["depth"], "vode29_nowaymo", "latest"),
        (RIGID_EF5, "kitti_raw", ["depth"], "vode30_standard", "latest"),
        (RIGID_EF5, "kitti_raw", ["depth"], "vode30_t2_comb", "latest"),
        (RIGID_EF5, "kitti_raw", ["depth"], "vode30_t2_moa", "latest"),
        (RIGID_EF5, "kitti_raw", ["depth"], "vode30_t2_t2", "latest"),
        (RIGID_EF5, "kitti_raw", ["depth"], "vode31_t2_comb_nopt", "latest"),
    ]

    TEST_PLAN = TEST_PLAN_HIGH if FixedOptions.HIGH_RES else TEST_PLAN_LOW

    """
    other options
    """
    ENABLE_SHAPE_DECOR = False
    LOG_LOSS = True
    TRAIN_MODE = ["eager", "graph", "distributed"][1]

    # download disparities_city2eigen_resnet.zip from http://visual.cs.ucl.ac.uk/pubs/monoDepth/results/
    MONODEPTH1_FILE = "/media/ri-bear/IntHDD/vode_data/vode_0103/othermethod/monodepth/disparities_city2eigen_resnet/disparities_pp.npy"
    # download mono+stereo 1024x320 from https://github.com/nianticlabs/monodepth2#-precomputed-results
    MONODEPTH2_FILE = "/media/ri-bear/IntHDD/vode_data/vode_0103/othermethod/monodepth2/mono+stereo_1024x320_eigen.npy"
    RAW_IMAGE_RES = {"kitti_raw": (375, 1242)}

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
        if code == "H":
            return imsize[0] // scale_div
        elif code == "W":
            return imsize[1] // scale_div
        elif code == "HW":
            return imsize
        elif code == "WH":
            return imsize[1] // scale_div, imsize[0] // scale_div
        elif code == "HWC":
            return imsize[0] // scale_div, imsize[1] // scale_div, 3
        elif code == "SHW":
            return cls.SNIPPET_LEN, imsize[0] // scale_div, imsize[1] // scale_div
        elif code == "SHWC":
            return cls.SNIPPET_LEN, imsize[0] // scale_div, imsize[1] // scale_div, 3
        elif code == "BSHWC":
            return cls.BATCH_SIZE, cls.SNIPPET_LEN, imsize[0] // scale_div, imsize[1] // scale_div, 3
        elif code == "RSHWC":
            return cls.PER_REPLICA_BATCH, cls.SNIPPET_LEN, imsize[0] // scale_div, imsize[1] // scale_div, 3
        else:
            assert 0, f"Invalid code: {code}"


opts = VodeOptions()
print(f"[config] ckpt path: {opts.DATAPATH_CKP}, nets: {opts.JOINT_NET}")

np.set_printoptions(precision=4, suppress=True, linewidth=150)