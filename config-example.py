import os.path as op


class VodeOptions:
    def __init__(self):
        self.DATASET = None
        self.SNIPPET_LEN = 5
        self.IM_WIDTH = 416
        self.IM_HEIGHT = 128
        self.BATCH_SIZE = 8
        self.EPOCHS = 100
        self.MIN_DEPTH = 1e-3
        self.MAX_DEPTH = 80
        self.ENABLE_SHAPE_DECOR = False
        self.SMOOTH_WEIGHT = 0.5

        self.DATAPATH = "/media/ian/IanPrivatePP/Datasets/vode_data"
        assert(op.isdir(self.DATAPATH))
        self.DATAPATH_SRC = op.join(self.DATAPATH, "srcdata")
        self.DATAPATH_TFR = op.join(self.DATAPATH, "tfrecords")
        self.DATAPATH_CKP = op.join(self.DATAPATH, "checkpts")
        self.DATAPATH_LOG = op.join(self.DATAPATH, "log")
        self.DATAPATH_PRD = op.join(self.DATAPATH, "prediction")
        self.DATAPATH_EVL = op.join(self.DATAPATH, "evaluation")


class KittiOptions(VodeOptions):
    def __init__(self):
        super().__init__()
        self.DATASET = "kitti_raw"
        self.KITTI_RAW_PATH = "/media/ian/IanPrivatePP/Datasets/kitti_raw_data"
        self.KITTI_ODOM_PATH = "/media/ian/IanPrivatePP/Datasets/kitti_odometry"
        if not op.isdir(self.KITTI_RAW_PATH):
            print("===== WARNING: kitti raw data path does NOT exists")
        if not op.isdir(self.KITTI_ODOM_PATH):
            print("===== WARNING: kitti odom data path does NOT exists")

    def get_dataset_path(self, dataset=None):
        if dataset is None:
            dataset = self.DATASET

        if dataset == "kitti_raw":
            return self.KITTI_RAW_PATH
        elif dataset == "kitti_odom":
            return self.KITTI_ODOM_PATH
        else:
            raise ValueError()


opts = KittiOptions()
