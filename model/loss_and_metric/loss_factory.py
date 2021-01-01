import tensorflow as tf
from config import opts
import model.loss_and_metric.losses as lm


def loss_factory(dataset_cfg, loss_weights, scale_weights, stereo=opts.STEREO,
                 weights_to_regularize=None, batch_size=opts.BATCH_SIZE):
    scale_weights = tf.convert_to_tensor(scale_weights.reshape(scale_weights.shape[0], 1), dtype=tf.float32)
    loss_pool = {
        "L1": lm.PhotometricLossMultiScale("L1", scale_weights),
        "L1_R": lm.PhotometricLossMultiScale("L1", scale_weights, key_suffix="_R"),
        "SSIM": lm.PhotometricLossMultiScale("SSIM", scale_weights),
        "SSIM_R": lm.PhotometricLossMultiScale("SSIM", scale_weights, key_suffix="_R"),

        "md2L1": lm.MonoDepth2LossMultiScale("L1", scale_weights),
        "md2L1_R": lm.MonoDepth2LossMultiScale("L1", scale_weights, key_suffix="_R"),
        "md2SSIM": lm.MonoDepth2LossMultiScale("SSIM", scale_weights),
        "md2SSIM_R": lm.MonoDepth2LossMultiScale("SSIM", scale_weights, key_suffix="_R"),

        "cmbL1": lm.CombinedLossMultiScale("L1", scale_weights),
        "cmbL1_R": lm.CombinedLossMultiScale("L1", scale_weights, key_suffix="_R"),
        "cmbSSIM": lm.CombinedLossMultiScale("SSIM", scale_weights),
        "cmbSSIM_R": lm.CombinedLossMultiScale("SSIM", scale_weights, key_suffix="_R"),

        "smoothe": lm.SmoothenessLossMultiScale(scale_weights),
        "smoothe_R": lm.SmoothenessLossMultiScale(scale_weights, key_suffix="_R"),
        "stereoL1": lm.StereoDepthLoss("L1", scale_weights),
        "stereoSSIM": lm.StereoDepthLoss("SSIM", scale_weights),
        "stereoPose": lm.StereoPoseLoss(),
        "flowL2": lm.FlowWarpLossMultiScale("L2", scale_weights),
        "flowL2_R": lm.FlowWarpLossMultiScale("L2", scale_weights, key_suffix="_R"),
        "flow_reg": lm.L2Regularizer(weights_to_regularize),
    }
    losses = dict()
    weights = dict()

    for name, weight in loss_weights.items():
        if weight == 0.:
            continue
        if not check_loss_dependency(name, dataset_cfg):
            continue
        losses[name] = loss_pool[name]
        weights[name] = weight

    print("[loss_factory] loss weights:", weights)
    print("[loss_factory] scale weights:", scale_weights[:, 0])
    return lm.TotalLoss(losses, weights, stereo, batch_size)


def check_loss_dependency(loss_key, dataset_cfg):
    loss_dependency = [(["L1", "SSIM", "smoothe", "flowL2", "flow_reg"],
                        ["image", "intrinsic"]),
                       (["L1_R", "SSIM_R", "smoothe_R", "flowL2_R"],
                        ["image_R", "intrinsic_R"]),
                       (["stereoL1", "stereoSSIM", "stereoPose"],
                        ["image", "intrinsic", "image_R", "intrinsic_R", "stereo_T_LR"])
                       ]
    # find dependency for loss_key
    dependents = []
    for loss_names, data_names in loss_dependency:
        if loss_key in loss_names:
            dependents = data_names
    # check if dependents are in dataset
    for dep in dependents:
        if dep not in dataset_cfg:
            print(f"[check_loss_dependency] {loss_key} loss is excluded because {dep} is NOT in dataset")
            return False

    return True
