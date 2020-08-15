import os.path as op
from config import opts
import tensorflow as tf
import json

import utils.util_funcs as uf
from utils.util_class import PathManager
from tfrecords.tfrecord_reader import TfrecordReader
from tfrecords.tfr_util import Serializer, show_example


def generate_validation_tfrecords(tfrpath):
    srcpath = check_source_path(tfrpath)
    if srcpath is None:
        return

    dataset = TfrecordReader(srcpath, shuffle=True, batch_size=1).get_dataset()
    with open(op.join(srcpath, "tfr_config.txt"), "r") as fr:
        config = json.load(fr)
    length = config["length"]
    serialize_example = Serializer()
    val_frames = opts.VALIDATION_FRAMES
    stride = max(min(length // val_frames, 10), 1)
    save_count = 0
    print("\n\n!!! Start create", op.basename(tfrpath))
    print(f"source length={length}, stride={stride}, val_frames={val_frames}")

    with PathManager([tfrpath]) as pm:
        outfile = f"{tfrpath}/validation.tfrecord"
        with tf.io.TFRecordWriter(outfile) as tfrwriter:
            for i, features in enumerate(dataset):
                if i % stride != 0:
                    continue

                example = convert_to_np(features)
                serialized = serialize_example(example)
                tfrwriter.write(serialized)
                save_count += 1
                uf.print_progress_status(f"[generate_validation] index: {i}, count: {save_count}/{val_frames}")
                if save_count % 100 == 10:
                    show_example(example, 100, True)
                elif save_count % 50 == 10:
                    show_example(example, 100)

            write_tfrecord_config(tfrpath, config, save_count)
        pm.set_ok()
    print("")


def check_source_path(tfrpath):
    if op.isdir(tfrpath.replace("_val", "_test")):
        return tfrpath.replace("_val", "_test")
    elif op.isdir(tfrpath.replace("_val", "_train")):
        return tfrpath.replace("_val", "_train")
    else:
        print("!!! NO source dataset for validation split:", tfrpath)
        return None


def convert_to_np(tffeats):
    npfeats = dict()
    for key, value in tffeats.items():
        if key.startswith("image5d"):
            continue

        if key.startswith("image"):
            value = uf.to_uint8_image(value)
        npfeats[key] = value[0].numpy()
    return npfeats


def write_tfrecord_config(tfrpath, cfg, example_count):
    cfg["length"] = example_count
    with open(op.join(tfrpath, "tfr_config.txt"), "w") as fr:
        json.dump(cfg, fr)

















