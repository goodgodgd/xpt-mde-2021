from model.build_model.model_base import BasicModel, NoResizingModel
from utils.util_class import WrongInputException


def model_factory(model_name, batch_size, image_shape, snippet_len):
    if model_name == "basic_model":
        return BasicModel(batch_size, image_shape, snippet_len).get_model()
    elif model_name == "no_resizing_model":
        return NoResizingModel(batch_size, image_shape, snippet_len).get_model()
    else:
        raise WrongInputException(f"{model_name} is NOT an available model name")
