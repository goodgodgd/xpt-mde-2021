import os.path as op

# TODO: Edit


class VodeOptions:
    def __init__(self):
        self.DATASET = "kitti_raw"
        self.SNIPPET_LEN = 5
        self.IM_WIDTH = 384
        self.IM_HEIGHT = 128
        self.BATCH_SIZE = 8
        self.EPOCHS = 100
        self.MIN_DEPTH = 1e-3
        self.MAX_DEPTH = 80
        self.ENABLE_SHAPE_DECOR = False
        self.SMOOTH_WEIGHT = 0.5

        self.DATAPATH = "/media/ian/IanPrivatePP/Datasets/vode_data_384"
        assert(op.isdir(self.DATAPATH))
        self.DATAPATH_SRC = op.join(self.DATAPATH, "srcdata")
        self.DATAPATH_TFR = op.join(self.DATAPATH, "tfrecords")
        self.DATAPATH_CKP = op.join(self.DATAPATH, "checkpts")
        self.DATAPATH_LOG = op.join(self.DATAPATH, "log")
        self.DATAPATH_PRD = op.join(self.DATAPATH, "prediction")
        self.DATAPATH_EVL = op.join(self.DATAPATH, "evaluation")


opts = VodeOptions()

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
