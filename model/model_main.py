import os
import os.path as op
import tensorflow as tf
import numpy as np

import settings
from config import opts
from tfrecords.tfrecord_reader import TfrecordReader
import utils.util_funcs as uf
from model.build_model.model_factory import ModelFactory
from model.model_util.augmentation import augmentation_factory
from model.loss_and_metric.loss_factory import loss_factory
from model.model_util.optimizers import optimizer_factory
import model.model_util.logger as log
from model.model_util.distributer import StrategyScope, StrategyDataset
import model.train_val as tv


def train_by_plan(plan):
    target_epoch = 0
    for net_names, dataset_name, epoch, learning_rate, loss_weights, save_ckpt in plan:
        target_epoch += epoch
        train(net_names, dataset_name, target_epoch, learning_rate, loss_weights, save_ckpt)


def train(net_names, dataset_name, target_epoch, learning_rate, loss_weights, save_ckpt):
    initial_epoch = uf.read_previous_epoch(opts.CKPT_NAME)
    if target_epoch <= initial_epoch:
        print(f"!! target_epoch {target_epoch} <= initial_epoch {initial_epoch}, no need to train")
        return

    set_configs()
    log.copy_or_check_same()
    dataset_train, tfr_config, train_steps = get_dataset(dataset_name, "train", True)
    dataset_val, _, val_steps = get_dataset(dataset_name, "val", False)
    model, augmenter, loss_object, optimizer = \
        create_training_parts(initial_epoch, tfr_config, learning_rate, loss_weights, net_names)
    trainer, validater = tv.train_val_factory(opts.TRAIN_MODE, model, loss_object,
                                              train_steps, opts.STEREO, augmenter, optimizer)

    print(f"\n\n========== START TRAINING ON {opts.CKPT_NAME} ==========")
    for epoch in range(initial_epoch, target_epoch):
        print(f"========== Start epoch: {epoch}/{target_epoch} ==========")
        result_train = trainer.run_an_epoch(dataset_train)
        result_val = validater.run_an_epoch(dataset_val)

        print("save intermediate results ...")
        log.save_reconstruction_samples(model, dataset_val, val_steps, epoch)
        log.save_log(epoch, dataset_name, result_train, result_val)
        save_model_weights(model, "latest")

    if save_ckpt:
        save_model_weights(model, f"ep{target_epoch:02}")


def set_configs():
    np.set_printoptions(precision=3, suppress=True)
    if not op.isdir(op.join(opts.DATAPATH_CKP, opts.CKPT_NAME)):
        os.makedirs(op.join(opts.DATAPATH_CKP, opts.CKPT_NAME), exist_ok=True)

    # set gpu configs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


@StrategyScope
def create_training_parts(initial_epoch, tfr_config, learning_rate, loss_weights, net_names=None):
    pretrained_weight = (initial_epoch == 0) and opts.PRETRAINED_WEIGHT
    model = ModelFactory(tfr_config, net_names=net_names, global_batch=opts.BATCH_SIZE, pretrained_weight=pretrained_weight).get_model()
    model = try_load_weights(model)
    # during joint training, flownet is frozen
    if ("depth" in net_names) and ("flow" in net_names):
        model.set_trainable("flownet", False)

    # model.compile(optimizer='sgd', loss='mean_absolute_error')
    augmenter = augmentation_factory(opts.AUGMENT_PROBS)
    loss_object = loss_factory(tfr_config, loss_weights, weights_to_regularize=model.weights_to_regularize())
    optimizer = optimizer_factory(opts.OPTIMIZER, learning_rate, initial_epoch)
    return model, augmenter, loss_object, optimizer


def try_load_weights(model, weights_suffix='latest'):
    if opts.CKPT_NAME:
        model_dir_path = op.join(opts.DATAPATH_CKP, opts.CKPT_NAME, "ckpt")
        if op.isdir(model_dir_path):
            model.load_weights(model_dir_path, weights_suffix)
        else:
            print("===== train from scratch:", model_dir_path)
    return model


@StrategyDataset
def get_dataset(dataset_name, split, shuffle, batch_size=opts.BATCH_SIZE):
    tfr_train_path = op.join(opts.DATAPATH_TFR, f"{dataset_name}_{split}")
    print("tfr path : ", tfr_train_path)
    assert op.isdir(tfr_train_path)
    tfr_reader = TfrecordReader(tfr_train_path, shuffle=shuffle, batch_size=batch_size)
    dataset = tfr_reader.get_dataset()
    tfr_config = tfr_reader.get_tfr_config()
    steps_per_epoch = uf.count_steps(tfr_train_path, batch_size)
    return dataset, tfr_config, steps_per_epoch


def save_model_weights(model, weights_suffix):
    """
    :param model: model wrapper instance
    :param weights_suffix: checkpoint name suffix
    """
    model_dir_path = op.join(opts.DATAPATH_CKP, opts.CKPT_NAME, "ckpt")
    if not op.isdir(model_dir_path):
        os.makedirs(model_dir_path, exist_ok=True)
    model.save_weights(model_dir_path, weights_suffix)


def predict_by_plan():
    for dataset_name, save_keys in opts.TEST_PLAN:
        predict(dataset_name, save_keys)


def predict(dataset_name, save_keys, weights_suffix="latest"):
    set_configs()
    dataset, tfr_config, steps = get_dataset(dataset_name, "test", True)
    model = ModelFactory(tfr_config).get_model()
    model = try_load_weights(model, weights_suffix)
    results = model.predict_dataset(dataset, save_keys, steps)
    # {pose: [N, numsrc, 6], depth: [N, height, width, 1]}
    for key, pred in results.items():
        print(f"[predict] key={key}, shape={pred.shape}")

    save_predictions(opts.CKPT_NAME, dataset_name, results)


def save_predictions(ckpt_name, dataset_name, results):
    pred_dir_path = op.join(opts.DATAPATH_PRD, ckpt_name)
    print(f"save predictions in {pred_dir_path})")
    os.makedirs(pred_dir_path, exist_ok=True)
    print(f"\tsave {dataset_name}.npy")
    np.savez(op.join(pred_dir_path, f"{dataset_name}.npz"), **results)


# ==================== tests ====================

def test_model_wrapper_output():
    ckpt_name = "vode1"
    test_dir_name = "kitti_raw_test"
    set_configs()
    model = ModelFactory().get_model()
    model = try_load_weights(model, ckpt_name)
    model.compile(optimizer="sgd", loss="mean_absolute_error")
    dataset = TfrecordReader(op.join(opts.DATAPATH_TFR, test_dir_name)).get_dataset()

    print("===== check model output shape")
    for features in dataset:
        preds = model(features)
        for i, (key, value) in enumerate(preds.items()):
            if isinstance(value, list):
                for k, val in enumerate(value):
                    print(f"check output[{i}]: key={key} [{k}], shape={val.get_shape().as_list()}, type={val.dtype}")
            else:
                print(f"check output[{i}]: key={key}, shape={value.get_shape().as_list()}, type={value.dtype}")
        break


def test_npz():
    src = {"A": np.ones((5, 5), dtype=np.int32), "B": np.zeros((3, 3), dtype=np.float32)}
    filename = "/home/ian/workspace/vode/test.npz"
    np.savez(filename, **src)
    dst = np.load(filename)
    print("keys:", dst.files)
    for key in dst.files:
        print(f"key={key}, value={dst[key]}")
    dst = {key: dst[key] for key in dst.files}
    print("to dict:", dst)


if __name__ == "__main__":
    train_by_plan(opts.PRE_TRAINING_PLAN)
    # train_by_plan(opts.FINE_TRAINING_PLAN)
    # predict_by_plan()

