from config import opts
import model.loss_and_metric.losses as lm


def loss_factory(loss_weights=opts.LOSS_WEIGHTS, stereo=opts.STEREO):
    losses = []
    weights = []
    if "L1" in loss_weights and loss_weights["L1"] > 0:
        losses.append(lm.PhotometricLossMultiScale("L1"))
        weights.append(loss_weights["L1"])
    if "L1_R" in loss_weights and loss_weights["L1_R"] > 0:
        losses.append(lm.PhotometricLossMultiScale("L1", key_suffix="_R"))
        weights.append(loss_weights["L1_R"])
    if "SSIM" in loss_weights and loss_weights["SSIM"] > 0:
        losses.append(lm.PhotometricLossMultiScale("SSIM"))
        weights.append(loss_weights["SSIM"])
    if "SSIM_R" in loss_weights and loss_weights["SSIM_R"] > 0:
        losses.append(lm.PhotometricLossMultiScale("SSIM", key_suffix="_R"))
        weights.append(loss_weights["SSIM_R"])
    if "smoothe" in loss_weights and loss_weights["smoothe"] > 0:
        losses.append(lm.SmoothenessLossMultiScale())
        weights.append(loss_weights["smoothe"])
    if "smoothe_R" in loss_weights and loss_weights["smoothe_R"] > 0:
        losses.append(lm.SmoothenessLossMultiScale(key_suffix="_R"))
        weights.append(loss_weights["smoothe_R"])
    if "stereo" in loss_weights and loss_weights["stereo"] > 0:
        losses.append(lm.StereoDepthLoss("L1"))
        weights.append(loss_weights["stereo"])

    return lm.TotalLoss(losses, weights, stereo)
