from config import opts
from model.loss_and_metric.losses import TotalLoss, PhotometricLossL1MultiScale, SmoothenessLossMultiScale
from utils.util_class import WrongInputException


def loss_factory(photo_loss_type=opts.PHOTO_LOSS, smoothness_weight=opts.SMOOTH_WEIGHT):
    losses = []
    if photo_loss_type == "L1":
        losses.append((PhotometricLossL1MultiScale(), 1.))
    else:
        WrongInputException(f"{photo_loss_type} is NOT an available photometric loss type")

    if smoothness_weight > 0:
        losses.append((SmoothenessLossMultiScale(), smoothness_weight))

    return TotalLoss(losses)
