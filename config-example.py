import os.path as op


class PriorOptions:
    def __init__(self):
        self.DATASET = "kitti_raw"
        self.SNIPPET_LEN = 5
        self.DATAPATH = "/media/ian/IanPrivatePP/Datasets/vode_data"
        self.IM_WIDTH = 416
        self.IM_HEIGHT = 128
        assert(op.isdir(self.DATAPATH))
        self.DATAPATH_SRC = op.join(self.DATAPATH, "srcdata")
        self.DATAPATH_TFR = op.join(self.DATAPATH, "tfrecords")


class VodeOptions(PriorOptions):
    def __init__(self):
        super().__init__()
        self.KITTI_RAW_PATH = "/media/ian/IanPrivatePP/Datasets/kitti_raw_data"
        self.KITTI_ODOM_PATH = "/media/ian/IanPrivatePP/Datasets/kitti_odometry"
        assert (op.isdir(self.KITTI_RAW_PATH))
        assert (op.isdir(self.KITTI_ODOM_PATH))

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
