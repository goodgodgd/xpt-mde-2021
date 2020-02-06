from config import opts
from model.build_model.model_base import BasicModel, NoResizingModel
from utils.util_class import WrongInputException

from model.build_model.pretrained_models import PretrainedModel

PRETRAINED_MODELS = ["MobileNetV2", "NASNetMobile", "DenseNet121", "VGG16", "Xception", "ResNet50V2", "NASNetLarge"]


def model_factory(model_type=opts.MODEL_TYPE,
                  batch_size=opts.BATCH_SIZE,
                  image_shape=(opts.IM_HEIGHT, opts.IM_WIDTH, 3),
                  snippet_len=opts.SNIPPET_LEN,
                  pretrained_weight=opts.PRETRAINED_WEIGHT):
    if model_type == "basic_model":
        return BasicModel(batch_size, image_shape, snippet_len).get_model()
    elif model_type == "no_resizing_model":
        return NoResizingModel(batch_size, image_shape, snippet_len).get_model()
    elif model_type in PRETRAINED_MODELS:
        return PretrainedModel(batch_size, image_shape, snippet_len, model_type, pretrained_weight).get_model()
    else:
        raise WrongInputException(f"{model_type} is NOT an available model name")
