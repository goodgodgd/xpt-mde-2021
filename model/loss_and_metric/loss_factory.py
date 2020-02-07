from config import opts
import model.loss_and_metric.losses as lm


def loss_factory(loss_weights=opts.LOSS_WEIGHTS):
    losses = []
    weights = []
    if "L1" in loss_weights and loss_weights["L1"] > 0:
        losses.append(lm.PhotometricLossL1MultiScale())
        weights.append(loss_weights["L1"])
    if "SSIM" in loss_weights and loss_weights["SSIM"] > 0:
        losses.append(lm.PhotometricLossSSIMMultiScale())
        weights.append(loss_weights["SSIM"])
    if "smootheness" in loss_weights and loss_weights["smootheness"] > 0:
        losses.append(lm.SmoothenessLossMultiScale())
        weights.append(loss_weights["smootheness"])

    return lm.TotalLoss(losses, weights)
