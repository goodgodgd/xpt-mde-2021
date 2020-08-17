from config import opts
import model.loss_and_metric.losses as lm


def loss_factory(dataset_cfg, loss_weights, stereo=opts.STEREO,
                 weights_to_regularize=None, batch_size=opts.BATCH_SIZE):
    loss_pool = {"L1": lm.PhotometricLossMultiScale("L1"),
                 "L1_R": lm.PhotometricLossMultiScale("L1", key_suffix="_R"),
                 "md2": lm.MonoDepth2LossMultiScale("L1"),
                 "md2_R": lm.MonoDepth2LossMultiScale("L1", key_suffix="_R"),
                 "SSIM": lm.PhotometricLossMultiScale("SSIM"),
                 "SSIM_R": lm.PhotometricLossMultiScale("SSIM", key_suffix="_R"),
                 "smoothe": lm.SmoothenessLossMultiScale(),
                 "smoothe_R": lm.SmoothenessLossMultiScale(key_suffix="_R"),
                 "stereoL1": lm.StereoDepthLoss("L1"),
                 "stereoSSIM": lm.StereoDepthLoss("SSIM"),
                 "stereoPose": lm.StereoPoseLoss(),
                 "flowL2": lm.FlowWarpLossMultiScale("L2"),
                 "flowL2_R": lm.FlowWarpLossMultiScale("L2", key_suffix="_R"),
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
