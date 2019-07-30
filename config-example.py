import os.path as op


class PriorOptions:
    def __init__(self):
        self.DATASET = "kitti_raw"
        self.SNIPPET_LEN = 5
        self.DATA_PATH = "/media/ian/iandata/vode_data"
        self.IM_WIDTH = 416
        self.IM_HEIGHT = 128


class VodeOptions(PriorOptions):
    def __init__(self):
        super().__init__()
        if self.DATASET == "kitti_raw":
            self.RAW_DATASET_PATH = "/media/ian/iandata/datasets/kitti_raw_data"
        elif self.DATASET == "kitti_odom":
            self.RAW_DATASET_PATH = "/media/ian/iandata/datasets/kitti_odometry/sequences"

        self.SNIPPET_PATH = op.join(self.DATA_PATH, "snippets", self.DATASET)


opts = VodeOptions()

