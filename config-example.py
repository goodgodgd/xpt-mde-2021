import os.path as op


class PriorOptions:
    def __init__(self):
        self.KITTI_RAW_PATH = "/media/ian/IanStudyPP/datasets/kitti_raw_data"
        self.KITTI_ODOM_PATH = "/media/ian/IanStudyPP/datasets/kitti_odometry"
        if not op.isdir(self.KITTI_RAW_PATH):
            print("===== WARNING: kitti raw data path does NOT exists")
        if not op.isdir(self.KITTI_ODOM_PATH):
            print("===== WARNING: kitti odom data path does NOT exists")

        self.DATAPATH = "/media/ian/IanStudyPP/paperdata/vode_data"
        assert(op.isdir(self.DATAPATH))

        self.DATASET = "kitti_raw"
        self.SNIPPET_LEN = 5
        self.IM_WIDTH = 416
        self.IM_HEIGHT = 128
        self.BATCH_SIZE = 8


class VodeOptions(PriorOptions):
    def __init__(self):
        super().__init__()
        self.DATAPATH_SRC = op.join(self.DATAPATH, "srcdata")
        self.DATAPATH_TFR = op.join(self.DATAPATH, "tfrecords")

    def get_dataset_path(self, dataset=None):
        if dataset is None:
            if self.DATASET == "kitti_raw":
                return self.KITTI_RAW_PATH
            elif self.DATASET == "kitti_odom":
                return self.KITTI_ODOM_PATH
        elif dataset == "kitti_raw":
            return self.KITTI_RAW_PATH
        elif dataset == "kitti_odom":
            return self.KITTI_ODOM_PATH
        else:
            raise ValueError()


opts = VodeOptions()
