import os
import os.path as op
from glob import glob
import numpy as np

import settings
from config import opts, get_raw_data_path
from utils.util_class import PathManager
from tfrecords.tfrecord_writer import TfrecordMaker


def convert_to_tfrecords():
    src_paths = glob(op.join(opts.DATAPATH_SRC, "*"))
    src_paths = [path for path in src_paths if op.isdir(path)]
    print("[convert_to_tfrecords] top paths:", src_paths)
    for srcpath in src_paths:
        tfrpath = op.join(opts.DATAPATH_TFR, op.basename(srcpath))
        if op.isdir(tfrpath):
            print("[convert_to_tfrecords] tfrecord already created in", tfrpath)
        else:
            with PathManager([tfrpath]) as pm:
                os.makedirs(tfrpath, exist_ok=True)
                tfrmaker = TfrecordMaker(srcpath, tfrpath, opts.STEREO,
                                         (opts.SNIPPET_LEN, opts.IM_HEIGHT, opts.IM_WIDTH, 3),
                                         opts.LIMIT_FRAMES, opts.SHUFFLE_TFRECORD_INPUT)
                tfrmaker.make()
                # if set_ok() was NOT excuted, the generated path is removed
                pm.set_ok()


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    convert_to_tfrecords()
