import os
import os.path as op
from glob import glob
import numpy as np

import settings
from config import opts
from tfrecords.tfrecord_writer import TfrecordMaker


def convert_to_tfrecords():
    src_paths = glob(op.join(opts.DATAPATH_SRC, "*"))
    src_paths = [path for path in src_paths if op.isdir(path)]
    print("[convert_to_tfrecords] top paths:", src_paths)

    for srcpath in src_paths:
        tfrpath = op.join(opts.DATAPATH_TFR, op.basename(srcpath))
        os.makedirs(tfrpath, exist_ok=True)
        tfrmaker = TfrecordMaker(srcpath, tfrpath)
        tfrmaker.make()


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    convert_to_tfrecords()
