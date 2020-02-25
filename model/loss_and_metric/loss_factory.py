from config import opts
import model.loss_and_metric.losses as lm


def loss_factory(loss_weights=opts.LOSS_WEIGHTS, stereo=opts.STEREO):
    losses = []
    weights = []
    loss_box = {"L1": lm.PhotometricLossMultiScale("L1"),
                "L1_R": lm.PhotometricLossMultiScale("L1", key_suffix="_R"),
                "SSIM": lm.PhotometricLossMultiScale("SSIM"),
                "SSIM_R": lm.PhotometricLossMultiScale("SSIM", key_suffix="_R"),
                "smoothe": lm.SmoothenessLossMultiScale(),
                "smoothe_R": lm.SmoothenessLossMultiScale(key_suffix="_R"),
                "stereo": lm.StereoDepthLoss("L1"),
                }

    for name, loss_w in loss_weights.items():
        if loss_w == 0.:
            continue

        losses.append(loss_box[name])
        weights.append(loss_w)

    return lm.TotalLoss(losses, weights, stereo)
