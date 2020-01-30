from tensorflow.keras import layers

from utils.util_class import WrongInputException
from model.synthesize.synthesize_base import SynthesizeBatchBasic, SynthesizeMultiScale


def synthesizer_factory(name):
    if name == "synthesize_basic":
        return SynthesizeBatchBasic()
    elif name == "synthesize_multi_scale":
        return SynthesizeMultiScale()
    else:
        raise WrongInputException(f"{name} is NOT an available synthesizer name")
