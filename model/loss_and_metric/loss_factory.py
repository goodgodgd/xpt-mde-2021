from config import opts
import model.loss_and_metric.losses as lm


def loss_factory(loss_weights=opts.LOSS_WEIGHTS, stereo=opts.STEREO,
                 weights_to_regularize=None, batch_size=opts.BATCH_SIZE):
    loss_pool = {"L1": lm.PhotometricLossMultiScale("L1"),
                 "L1_R": lm.PhotometricLossMultiScale("L1", key_suffix="_R"),
                 "SSIM": lm.PhotometricLossMultiScale("SSIM"),
                 "SSIM_R": lm.PhotometricLossMultiScale("SSIM", key_suffix="_R"),
                 "FW_L2": lm.FlowWarpLossMultiScale("L2"),
                 "FW_L2_R": lm.FlowWarpLossMultiScale("L2", key_suffix="_R"),
                 "smoothe": lm.SmoothenessLossMultiScale(),
                 "smoothe_R": lm.SmoothenessLossMultiScale(key_suffix="_R"),
                 "stereo_L1": lm.StereoDepthLoss("L1"),
                 "stereo_pose": lm.StereoPoseLoss(),
                 "FW_L2_regular": lm.L2Regularizer(weights_to_regularize),
                 }

    losses = dict()
    weights = dict()
    for name, weight in loss_weights.items():
        if weight == 0.:
            continue
        losses[name] = loss_pool[name]
        weights[name] = weight

    print("[loss_factory] loss weights:", weights)
    return lm.TotalLoss(losses, weights, stereo, batch_size)
