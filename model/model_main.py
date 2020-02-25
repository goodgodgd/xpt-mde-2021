import os
import os.path as op
import tensorflow as tf
import numpy as np

import settings
from config import opts
from tfrecords.tfrecord_reader import TfrecordGenerator
import utils.util_funcs as uf
from model.build_model.model_factory import ModelFactory
from model.optimizers import optimizer_factory
import model.logger as log
import model.train_val as tv


def train():
    initial_epoch = uf.read_previous_epoch(opts.CKPT_NAME)
    if opts.EPOCHS <= initial_epoch:
        print(f"!! final_epoch {opts.EPOCHS} <= initial_epoch {initial_epoch}, no need to train")
        return

    set_configs()
    log.copy_or_check_same()
    pretrained_weight = (initial_epoch == 0) and opts.PRETRAINED_WEIGHT
    model = ModelFactory(pretrained_weight=pretrained_weight).get_model()
    model = try_load_weights(model, opts.CKPT_NAME)
    model.compile(optimizer='sgd', loss='mean_absolute_error')

    # TODO WARNING! using "test" split for training dataset is just to check training process
    dataset_train, train_steps = get_dataset(opts.DATASET, "test", True)
    dataset_val, val_steps = get_dataset(opts.DATASET, "test", False)
    optimizer = optimizer_factory("adam_constant", opts.LEARNING_RATE, initial_epoch)
    trainer_graph = tv.ModelTrainerGraph(train_steps, opts.STEREO, optimizer)
    validater_graph = tv.ModelValidaterGraph(val_steps, opts.STEREO)

    print(f"\n\n========== START TRAINING ON {opts.CKPT_NAME} ==========")
    for epoch in range(initial_epoch, opts.EPOCHS):
        print(f"========== Start epoch: {epoch}/{opts.EPOCHS} ==========")

        result_train, depth_train = trainer_graph.run_an_epoch(model, dataset_train)
        result_val, depth_val = validater_graph.run_an_epoch(model, dataset_val)

        # if epoch % 5 == 0:
        print("save intermediate results ...")
        log.save_reconstruction_samples(model, dataset_val, epoch)
        log.save_loss_scales(model, dataset_val, val_steps, opts.STEREO)
        save_model(model, result_val[0])
        log.save_log(epoch, result_train, result_val, depth_train, depth_val)


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


def try_load_weights(model, weight_name='latest.h5'):
    if opts.CKPT_NAME:
        model_file_path = op.join(opts.DATAPATH_CKP, opts.CKPT_NAME, weight_name)
        if op.isfile(model_file_path):
            print("===== load model weights", model_file_path)
            model.load_weights(model_file_path)
        else:
            print("===== train from scratch", model_file_path)
    return model


def get_dataset(dataset_name, split, shuffle, batch_size=opts.BATCH_SIZE):
    tfr_train_path = op.join(opts.DATAPATH_TFR, f"{dataset_name}_{split}")
    assert op.isdir(tfr_train_path)
    dataset = TfrecordGenerator(tfr_train_path, shuffle=shuffle, batch_size=batch_size).get_generator()
    steps_per_epoch = uf.count_steps(tfr_train_path, batch_size)
    return dataset, steps_per_epoch


def save_model(model, val_loss):
    """
    :param model: nn model object
    :param val_loss: current validation loss
    """
    # save the latest model
    save_model_weights(model, 'latest')
    # save the best model (function static variable)
    save_model.best = getattr(save_model, 'best', 10000)
    if val_loss < save_model.best:
        save_model_weights(model, 'best')
        save_model.best = val_loss


def save_model_weights(model, weights_name):
    model_dir_path = op.join(opts.DATAPATH_CKP, opts.CKPT_NAME)
    if not op.isdir(model_dir_path):
        os.makedirs(model_dir_path, exist_ok=True)
    model.save_weights(model_dir_path, weights_name)


def predict(weight_name="latest.h5"):
    set_configs()
    batch_size = 1
    input_shape = (batch_size, opts.SNIPPET_LEN, opts.IM_HEIGHT, opts.IM_WIDTH, 3)
    model = ModelFactory(input_shape=input_shape).get_model()
    model = try_load_weights(model, weight_name)
    model.compile(optimizer="sgd", loss="mean_absolute_error")

    dataset, steps = get_dataset(opts.DATASET, "test", False, batch_size)
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


def test_train_disparity():
    """
    DEPRECATED: this function can be executed only when getting model_builder.py back to the below commit
    commit: 68612cb3600cfc934d8f26396b51aba0622ba357
    """
    ckpt_name = "vode1"
    check_epochs = [1] + [5]*5
    dst_epoch = 0
    for i, epochs in enumerate(check_epochs):
        dst_epoch += epochs
        # train a few epochs
        train()
        check_disparity(ckpt_name, "kitti_raw_test")


def check_disparity(ckpt_name, test_dir_name):
    """
    DEPRECATED: this function can be executed only when getting model_builder.py back to the below commit
    commit: 68612cb3600cfc934d8f26396b51aba0622ba357
    """
    set_configs()
    model = ModelFactory().get_model()
    model = try_load_weights(model, ckpt_name)
    model.compile(optimizer="sgd", loss="mean_absolute_error")
    dataset = TfrecordGenerator(op.join(opts.DATAPATH_TFR, test_dir_name)).get_generator()

    for i, features in enumerate(dataset):
        predictions = model.predict(features['image'])

        pred_disp_s1 = predictions[0]
        pred_disp_s4 = predictions[2]
        pred_pose = predictions[4]

        for k, conv in enumerate(predictions[5:]):
            print(f"conv{k} stats", np.mean(conv), np.std(conv), np.quantile(conv, [0, 0.2, 0.5, 0.8, 1.0]))

        batch, height, width, _ = pred_disp_s1.shape
        view_y, view_x = int(height * 0.3), int(width * 0.3)
        print("view pixel", view_y, view_x)
        print(f"disp scale 1, {pred_disp_s1.shape}\n", pred_disp_s1[0, view_y:view_y+50:10, view_x:view_x+100:10, 0])
        batch, height, width, _ = pred_disp_s4.shape
        view_y, view_x = int(height * 0.5), int(width * 0.3)
        print(f"disp scale 1/4, {pred_disp_s4.shape}\n", pred_disp_s4[0, view_y:view_y+50:10, view_x:view_x+100:10, 0])
        print("pose\n", pred_pose[0, 0])
        if i > 5:
            break


if __name__ == "__main__":
    for epoch in range(0, 51, 15):
        opts.EPOCHS = epoch
        train()
    # predict()
    # test_model_wrapper_output()
