import numpy as np
import tensorflow as tf
import os.path as op

from config import opts
import utils.util_funcs as uf
import utils.convert_pose as cp


class ModelWrapper:
    def __init__(self, models):
        self.models = models

    def __call__(self, features):
        predictions = self.predict_batch(features)
        return predictions

    def predict_dataset(self, dataset, save_keys, total_steps):
        outputs = {key: [] for key in save_keys}
        outputs.update({key + "_gt": [] for key in save_keys})
        for step, features in enumerate(dataset):
            predictions = self.predict_batch(features)
            outputs = self.append_outputs(features, predictions, outputs)
            uf.print_progress_status(f"Progress: {step} / {total_steps}")

        print("")
        # concatenate batch outputs along batch axis
        results = {}
        for key, data in outputs.items():
            if data:
                results[key] = np.concatenate(data, axis=0)
        return results

    def predict_batch(self, features, suffix=""):
        predictions = dict()
        for netname, model in self.models.items():
            pred = model(features["image5d" + suffix])
            predictions.update(pred)

        if "depth_ms" in predictions:
            predictions["disp_ms"] = uf.safe_reciprocal_number_ms(predictions["depth_ms"])

        predictions = {key + suffix: value for key, value in predictions.items()}
        return predictions

    def append_outputs(self, features, predictions, outputs, suffix=""):
        if "pose" + suffix in outputs:
            # [batch, numsrc, 6]
            pose_gt = features["pose_gt" + suffix]
            outputs["pose_gt" + suffix].append(pose_gt)
            outputs["pose" + suffix].append(predictions["pose" + suffix])
        # only the highest resolution ouput is used for evaluation
        if "depth" + suffix in outputs:
            # [batch, height, width, 1]
            depth_gt = features["depth_gt" + suffix]
            outputs["depth_gt" + suffix].append(depth_gt)
            depth_ms = predictions["depth_ms" + suffix]
            outputs["depth" + suffix].append(depth_ms[0])
        if "flow" + suffix in outputs:
            # [batch, numsrc, height, width, 2]
            # TODO: add "flow_gt" to tfrecords
            # flow_gt = features["flow_gt" + suffix]
            # outputs["flow_gt" + suffix].append(flow_gt)
            flow_ms = predictions["flow_ms" + suffix]
            outputs["flow" + suffix].append(flow_ms[0])
        return outputs

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
        predictions = self.predict_batch(features)
        preds_right = self.predict_batch(features, "_R")
        predictions.update(preds_right)
        return predictions


class StereoPoseModelWrapper(StereoModelWrapper):
    def __init__(self, models):
        super().__init__(models)

    def __call__(self, features):
        predictions = self.predict_batch(features)
        preds_right = self.predict_batch(features, "_R")
        predictions.update(preds_right)
        if "posenet" in self.models:
            stereo_pose = self.predict_stereo_pose(features)
            predictions.update(stereo_pose)
        return predictions

    def predict_stereo_pose(self, features):
        # predicts stereo extrinsic in both directions: left to right, right to left
        posenet = self.models["posenet"]
        left_target = features["image5d"][:, -1]
        right_target = features["image5d_R"][:, -1]
        numsrc = opts.SNIPPET_LEN - 1
        lr_input = tf.stack([right_target] * numsrc + [left_target], axis=1)
        rl_input = tf.stack([left_target] * numsrc + [right_target], axis=1)
        # lr_input = layers.concatenate([right_target] * numsrc + [left_target], axis=1)
        # rl_input = layers.concatenate([left_target] * numsrc + [right_target], axis=1)

        # pose that transforms points from right to left (T_LR)
        pose_lr = posenet(lr_input)
        # pose that transforms points from left to right (T_RL)
        pose_rl = posenet(rl_input)
        outputs = {"pose_LR": pose_lr["pose"], "pose_RL": pose_rl["pose"]}
        return outputs
