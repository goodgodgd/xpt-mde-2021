import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os.path as op

from config import opts
import utils.util_funcs as uf


class ModelWrapper:
    """
    tf.keras.Model output formats according to prediction methods
    1) preds = model(image_tensor) -> dict('disp_ms': disp_ms, 'pose': pose)
        disp_ms: list of [batch, height/scale, width/scale, 1]
        pose: [batch, num_src, 6]
    2) preds = model.predict(image_tensor) -> [disp_s1, disp_s2, disp_s4, disp_s8, pose]
    3) preds = model.predict({'image':, ...}) -> [disp_s1, disp_s2, disp_s4, disp_s8, pose]
    """
    def __init__(self, models):
        self.models = models

    def __call__(self, features):
        predictions = dict()
        for netname, model in self.models.items():
            pred = model(features["image"])
            predictions.update(pred)
        if "depth_ms" in predictions:
            predictions["disp_ms"] = uf.safe_reciprocal_number_ms(predictions["depth_ms"])
        return predictions

    def predict(self, dataset, total_steps):
        return self.predict_oneside(dataset, "image", total_steps)

    def predict_oneside(self, dataset, image_key, total_steps):
        print(f"===== start prediction from [{image_key}] key")
        predictions = {"depth": [], "pose": []}
        for step, features in enumerate(dataset):
            depth_ms = self.models["depthnet"](features[image_key])
            predictions["depth"].append(depth_ms[0])
            pose = self.models["posenet"](features[image_key])
            predictions["pose"].append(pose)
            uf.print_progress_status(f"Progress: {step} / {total_steps}")

        print("")
        predictions["depth"] = np.concatenate(predictions["depth"], axis=0)
        predictions["pose"] = np.concatenate(predictions["pose"], axis=0)
        return predictions

    def compile(self, optimizer="sgd", loss="mean_absolute_error"):
        for model in self.models.values():
            model.compile(optimizer=optimizer, loss=loss)

    def trainable_weights(self):
        train_weights = []
        for model in self.models.values():
            train_weights.extend(model.trainable_weights)
        return train_weights

    def weights_to_regularize(self):
        if "flownet" in self.models:
            return self.models["flownet"].trainable_weights
        else:
            return None

    def save_weights(self, ckpt_dir_path, suffix):
        for netname, model in self.models.items():
            save_path = op.join(ckpt_dir_path, f"{netname}_{suffix}.h5")
            model.save_weights(save_path)

    def load_weights(self, ckpt_dir_path, suffix):
        for netname in self.models.keys():
            ckpt_file = op.join(ckpt_dir_path, f"{netname}_{suffix}.h5")
            if op.isfile(ckpt_file):
                self.models[netname].load_weights(ckpt_file)
                print(f"===== {netname} weights loaded from", ckpt_file)
            else:
                print(f"===== Failed to load weights of {netname}, train from scratch ...")
                print(f"      tried to load file:", ckpt_file)

    def summary(self, **kwargs):
        for model in self.models.values():
            model.summary(**kwargs)

    def inputs(self):
        return [model.input for model in self.models.values()]

    def outputs(self):
        output_dict = dict()
        for model in self.models.values():
            output_dict.update(model.output)
        return output_dict

    def plot_model(self, dir_path):
        for netname, model in self.models.items():
            tf.keras.utils.plot_model(model, to_file=op.join(dir_path, netname + ".png"), show_shapes=True)


class StereoModelWrapper(ModelWrapper):
    def __init__(self, models):
        super().__init__(models)

    def __call__(self, features):
        predictions = dict()
        for netname, model in self.models.items():
            pred = model(features["image"])
            predictions.update(pred)
            preds_right = model(features["image_R"])
            preds_right = {key + "_R": value for key, value in preds_right.items()}
            predictions.update(preds_right)
        if "depth_ms" in predictions:
            predictions["disp_ms"] = uf.safe_reciprocal_number_ms(predictions["depth_ms"])
        if "depth_ms_R" in predictions:
            predictions["disp_ms_R"] = uf.safe_reciprocal_number_ms(predictions["depth_ms_R"])
        return predictions

    def predict(self, dataset, total_steps):
        predictions = self.predict_oneside(dataset, "image", total_steps)
        preds_right = self.predict_oneside(dataset, "image_R", total_steps)
        preds_right = {key + "_R": value for key, value in preds_right.items()}
        predictions.update(preds_right)
        return predictions


class StereoPoseModelWrapper(ModelWrapper):
    def __init__(self, models):
        super().__init__(models)

    def __call__(self, features):
        predictions = dict()
        for netname, model in self.models.items():
            pred = model(features["image"])
            predictions.update(pred)
            preds_right = model(features["image_R"])
            preds_right = {key + "_R": value for key, value in preds_right.items()}
            predictions.update(preds_right)

        print("!!! stereo wrapper model prediction keys:", predictions.keys())
        if "depth_ms" in predictions:
            predictions["disp_ms"] = uf.safe_reciprocal_number_ms(predictions["depth_ms"])
        if "depth_ms_R" in predictions:
            predictions["disp_ms_R"] = uf.safe_reciprocal_number_ms(predictions["depth_ms_R"])

        # predicts stereo extrinsic in both directions: left to right, right to left
        if "posenet" in self.models:
            posenet = self.models["posenet"]
            left_source, left_target = uf.split_into_source_and_target(features["image"])
            right_source, right_target = uf.split_into_source_and_target(features["image_R"])
            num_src = opts.SNIPPET_LEN - 1
            lr_input = layers.concatenate([right_target] * num_src + [left_target], axis=1)
            rl_input = layers.concatenate([left_target] * num_src + [right_target], axis=1)
            # pose that transforms points from right to left (T_LR)
            pose_lr = posenet(lr_input)
            # pose that transforms points from left to right (T_RL)
            pose_rl = posenet(rl_input)
            predictions["pose_LR"] = pose_lr["pose"]
            predictions["pose_RL"] = pose_rl["pose"]

        return predictions
