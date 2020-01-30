from config import opts
from model.build_model.model_base import BasicModel, NoResizingModel
from utils.util_class import WrongInputException


def model_factory(model_type=opts.MODEL_TYPE, batch_size=opts.BATCH_SIZE,
                  image_shape=(opts.IM_HEIGHT, opts.IM_WIDTH, 3), snippet_len=opts.SNIPPET_LEN):
    if model_type == "basic_model":
        return BasicModel(batch_size, image_shape, snippet_len).get_model()
    elif model_type == "no_resizing_model":
        return NoResizingModel(batch_size, image_shape, snippet_len).get_model()
    else:
        raise WrongInputException(f"{model_type} is NOT an available model name")
