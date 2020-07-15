from config import opts
import model.loss_and_metric.losses as lm


def loss_factory(loss_weights=opts.LOSS_WEIGHTS, stereo=opts.STEREO, weights_to_regularize=None):
    loss_pool = {"L1": lm.PhotometricLossMultiScale("L1"),
                 "L1_R": lm.PhotometricLossMultiScale("L1", key_suffix="_R"),
                 "SSIM": lm.PhotometricLossMultiScale("SSIM"),
                 "SSIM_R": lm.PhotometricLossMultiScale("SSIM", key_suffix="_R"),
                 "FW_L1": lm.FlowWarpLossMultiScale("L1"),
                 "FW_L1_R": lm.FlowWarpLossMultiScale("L1", key_suffix="_R"),
                 "FW_SSIM": lm.FlowWarpLossMultiScale("SSIM"),
                 "FW_SSIM_R": lm.FlowWarpLossMultiScale("SSIM", key_suffix="_R"),
                 "smoothe": lm.SmoothenessLossMultiScale(),
                 "smoothe_R": lm.SmoothenessLossMultiScale(key_suffix="_R"),
                 "stereo_L1": lm.StereoDepthLoss("L1"),
                 "stereo_pose": lm.StereoPoseLoss(),
                 "L2_regularizer": lm.L2Regularizer(weights_to_regularize),
                }

    losses = []
    weights = []
    for name, loss_w in loss_weights.items():
        if loss_w == 0.:
            continue
        losses.append(loss_pool[name])
        weights.append(loss_w)

    return lm.TotalLoss(losses, weights, stereo)
