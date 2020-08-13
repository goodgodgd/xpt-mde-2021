import os
import os.path as op
import tensorflow as tf
import numpy as np

import settings
from config import opts
from tfrecords.tfrecord_reader import TfrecordGenerator
import utils.util_funcs as uf
from model.build_model.model_factory import ModelFactory
from model.model_util.augmentation import augmentation_factory
from model.loss_and_metric.loss_factory import loss_factory
from model.model_util.optimizers import optimizer_factory
import model.model_util.logger as log
from model.model_util.distributer import StrategyScope, StrategyDataset
import model.train_val as tv


def train(final_epoch=opts.EPOCHS):
    initial_epoch = uf.read_previous_epoch(opts.CKPT_NAME)
    if final_epoch <= initial_epoch:
        print(f"!! final_epoch {final_epoch} <= initial_epoch {initial_epoch}, no need to train")
        return

    set_configs()
    log.copy_or_check_same()
    model, augmenter, loss_object, optimizer = create_training_parts(initial_epoch)

    # TODO WARNING! using "test" split for training dataset is just to check training process
    dataset_train, train_steps = get_dataset(opts.DATASET_TO_USE, "train", True)
    dataset_val, val_steps = get_dataset(opts.DATASET_TO_USE, "val", False)
    trainer, validater = tv.train_val_factory(opts.TRAIN_MODE, model, loss_object,
                                              train_steps, opts.STEREO, augmenter, optimizer)

    print(f"\n\n========== START TRAINING ON {opts.CKPT_NAME} ==========")
    for epoch in range(initial_epoch, final_epoch):
        print(f"========== Start epoch: {epoch}/{final_epoch} ==========")
        result_train = trainer.run_an_epoch(dataset_train)
        result_val = validater.run_an_epoch(dataset_val)

        print("save intermediate results ...")
        log.save_reconstruction_samples(model, dataset_val, val_steps, epoch)
        save_model(model, result_val)
        log.save_log(epoch, result_train, result_val)


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
def create_training_parts(initial_epoch):
    pretrained_weight = (initial_epoch == 0) and opts.PRETRAINED_WEIGHT
    model = ModelFactory(global_batch=opts.BATCH_SIZE, pretrained_weight=pretrained_weight).get_model()
    model = try_load_weights(model)
    model.compile(optimizer='sgd', loss='mean_absolute_error')
    augmenter = augmentation_factory(opts.AUGMENT_PROBS)
    loss_object = loss_factory(weights_to_regularize=model.weights_to_regularize())
    optimizer = optimizer_factory(opts.OPTIMIZER, opts.LEARNING_RATE, initial_epoch)
    return model, augmenter, loss_object, optimizer


def try_load_weights(model, weights_suffix='latest'):
    if opts.CKPT_NAME:
        model_dir_path = op.join(opts.DATAPATH_CKP, opts.CKPT_NAME)
        if op.isdir(model_dir_path):
            model.load_weights(model_dir_path, weights_suffix)
        else:
            print("===== train from scratch", model_dir_path)
    return model


@StrategyDataset
def get_dataset(dataset_name, split, shuffle):
    batch_size = opts.BATCH_SIZE
    tfr_train_path = op.join(opts.DATAPATH_TFR, f"{dataset_name}_{split}")
    assert op.isdir(tfr_train_path)
    dataset = TfrecordGenerator(tfr_train_path, shuffle=shuffle, batch_size=batch_size).get_generator()
    steps_per_epoch = uf.count_steps(tfr_train_path, batch_size)
    return dataset, steps_per_epoch


def save_model(model, results_val):
    """
    :param model: nn model object
    :param results_val: validatation results
    """
    val_loss = results_val['loss'].mean()
    # save the latest model
    save_model_weights(model, 'latest')
    # save the best model (function static variable)
    save_model.best = getattr(save_model, 'best', 10000)
    if val_loss < save_model.best:
        save_model_weights(model, 'best')
        save_model.best = val_loss


def save_model_weights(model, weights_suffix):
    model_dir_path = op.join(opts.DATAPATH_CKP, opts.CKPT_NAME)
    if not op.isdir(model_dir_path):
        os.makedirs(model_dir_path, exist_ok=True)
    model.save_weights(model_dir_path, weights_suffix)


def predict(weight_name="latest.h5"):
    set_configs()
    model = ModelFactory(global_batch=1).get_model()
    model = try_load_weights(model, weight_name)
    model.compile(optimizer="sgd", loss="mean_absolute_error")

    dataset, steps = get_dataset(opts.DATASET_TO_USE, "test", False)
    # [disp_s1, disp_s2, disp_s4, disp_s8, pose] = model.predict({"image": ...})
    # TODO: predict and collect outputs in for loop
    predictions = model.predict(dataset, steps)
    for key, pred in predictions.items():
        print(f"prediction: key={key}, shape={pred.shape}")

    save_predictions(opts.CKPT_NAME, predictions)


def save_predictions(ckpt_name, predictions):
    pred_dir_path = op.join(opts.DATAPATH_PRD, ckpt_name)
    print(f"save predictions in {pred_dir_path})")
    os.makedirs(pred_dir_path, exist_ok=True)
    for key, value in predictions.items():
        print(f"\tsave {key}.npy")
        np.save(op.join(pred_dir_path, f"{key}.npy"), value)


# ==================== tests ====================

def test_model_wrapper_output():
    ckpt_name = "vode1"
    test_dir_name = "kitti_raw_test"
    set_configs()
    model = ModelFactory().get_model()
    model = try_load_weights(model, ckpt_name)
    model.compile(optimizer="sgd", loss="mean_absolute_error")
    dataset = TfrecordGenerator(op.join(opts.DATAPATH_TFR, test_dir_name)).get_generator()

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


if __name__ == "__main__":
    reset_period = 15
    for epoch_ in range(reset_period, opts.EPOCHS, reset_period):
        train(epoch_)
    # predict()
    # test_model_wrapper_output()
