from tensorflow.keras import layers

from utils.util_class import WrongInputException
from utils.decorators import shape_check
from utils.convert_pose import pose_rvec2matr_batch
from model.synthesize.synthesize_base import SynthesizeBatchBasic


@shape_check
def synthesize_batch_multi_scale(src_img_stacked, intrinsic, pred_depth_ms, pred_pose, synth_type):
    """
    :param src_img_stacked: [batch, height*num_src, width, 3]
    :param intrinsic: [batch, 3, 3]
    :param pred_depth_ms: predicted depth in multi scale, list of [batch, height/scale, width/scale, 1]}
    :param pred_pose: predicted source pose in twist form [batch, num_src, 6]
    :return: reconstructed target view in multi scale, list of [batch, num_src, height/scale, width/scale, 3]}
    """
    # convert pose vector to transformation matrix
    poses_matr = layers.Lambda(lambda pose: pose_rvec2matr_batch(pose),
                               name="pose2matrix")(pred_pose)
    synth_images = []
    synthesizer = synthesizer_factory(synth_type)
    for depth_sc in pred_depth_ms:
        synth_image_sc = synthesizer(src_img_stacked, intrinsic, depth_sc, poses_matr)
        synth_images.append(synth_image_sc)
    return synth_images


def synthesizer_factory(name):
    if name == "synthesize_basic":
        return SynthesizeBatchBasic()
    else:
        raise WrongInputException(f"{name} is NOT an available synthesizer name")
