import os.path as op
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import settings
from config import opts
from model.synthesize_batch import synthesize_batch_multi_scale
import model.loss_and_metric as lm


def create_model():
    image_shape = (opts.IM_HEIGHT * opts.SNIPPET_LEN, opts.IM_WIDTH, 3)
    stacked_image = layers.Input(shape=image_shape, batch_size=opts.BATCH_SIZE, name="image")
    target_image = layers.Lambda(lambda image: extract_target_image(image), name="extract_target_image")(stacked_image)
    model = create_pred_model(stacked_image, target_image)
    return model


def extract_target_image(stacked_image):
    """
    :param stacked_image: [batch, snippet_len*height, width, 3]
    :return: target_image, [batch, height, width, 3]
    """
    batch, imheight, imwidth, _ = stacked_image.get_shape().as_list()
    imheight = int(imheight // opts.SNIPPET_LEN)
    target_image = tf.slice(stacked_image, (0, imheight*(opts.SNIPPET_LEN-1), 0, 0),
                            (-1, imheight, -1, -1))
    return target_image


def create_pred_model(stacked_image, target_image):
    """
    :param stacked_image: [batch, snippet_len*height, width, 3]
    :param target_image: [batch, height, width, 3]
    :return: prediction model
    """
    print(f"pred model input shapes: stacked_image {stacked_image.get_shape()}, "
          f"target_image {target_image.get_shape()}")
    pred_disps_ms = build_depth_estim_layers(target_image)
    pred_poses = build_visual_odom_layers(stacked_image)

    model_input = {"image": stacked_image}
    predictions = {"disp_ms": pred_disps_ms, "pose": pred_poses}
    model_pred = tf.keras.Model(model_input, predictions)
    return model_pred


def create_train_model(model_pred, target_image, intrinsic, depth_gt):
    """
    :param model_pred: pose and depth prediction model
    :param target_image: [batch, height, width, 3]
    :param intrinsic: camera projection matrix [batch, num_src, 3, 3]
    :param depth_gt: ground truth depth [batch, height, width, 1] or None
    :return: trainable model
    """
    # calculate loss and make model for training
    stacked_image = model_pred.input["image"]
    pred_disp_ms = model_pred.output["disp_ms"]
    pred_pose = model_pred.output["pose"]
    pred_depth_ms = disp_to_depth(pred_disp_ms)
    target_ms = multi_scale_like(target_image, pred_disp_ms)

    synth_target_ms = synthesize_batch_multi_scale(stacked_image, intrinsic,
                                                   pred_depth_ms, pred_pose)
    photo_loss = lm.photometric_loss_multi_scale(synth_target_ms, target_ms)
    height_orig = target_image.get_shape().as_list()[2]
    smooth_loss = lm.smootheness_loss_multi_scale(pred_disp_ms, target_ms, height_orig)
    loss = layers.Lambda(lambda losses: tf.add(losses[0], losses[1]), name="loss_out")\
                        ([photo_loss, smooth_loss])

    metric = layers.Lambda(lambda depths: lm.depth_error_metric(depths[0], depths[1]),
                           name="metric_out")([pred_depth_ms[0], depth_gt])

    inputs = {"image": stacked_image, "intrinsic": intrinsic, "depth_gt": depth_gt}
    outputs = {"loss_out": loss, "metric_out": metric}
    model_train = tf.keras.Model(inputs, outputs)
    return model_train


def disp_to_depth(disp_ms):
    target_ms = []
    for i, disp in enumerate(disp_ms):
        target = layers.Lambda(lambda dis: 1./dis, name=f"todepth_{i}")(disp)
        target_ms.append(target)
    return target_ms


def multi_scale_like(image, disp_ms):
    """
    :param image: [batch, height, width, 3]
    :param disp_ms: list of [batch, height/scale, width/scale, 1]
    :return: image_ms: list of [batch, height/scale, width/scale, 3]
    """
    image_ms = []
    for i, disp in enumerate(disp_ms):
        batch, height_sc, width_sc, _ = disp.get_shape().as_list()
        image_sc = layers.Lambda(lambda img: tf.image.resize(img, size=(height_sc, width_sc), method="bilinear"),
                                 name=f"target_resize_{i}")(image)
        image_ms.append(image_sc)
    return image_ms


# ==================== build DepthNet layers ====================
DISP_SCALING_VGG = 10


def build_depth_estim_layers(target_image):
    batch, imheight, imwidth, imchannel = target_image.get_shape().as_list()

    conv1 = convolution(target_image, 32, 7, strides=1, name="dp_conv1a")
    conv1 = convolution(conv1, 32, 7, strides=2, name="dp_conv1b")
    conv2 = convolution(conv1, 64, 5, strides=1, name="dp_conv2a")
    conv2 = convolution(conv2, 64, 5, strides=2, name="dp_conv2b")
    conv3 = convolution(conv2, 128, 3, strides=1, name="dp_conv3a")
    conv3 = convolution(conv3, 128, 3, strides=2, name="dp_conv3b")
    conv4 = convolution(conv3, 256, 3, strides=1, name="dp_conv4a")
    conv4 = convolution(conv4, 256, 3, strides=2, name="dp_conv4b")
    conv5 = convolution(conv4, 512, 3, strides=1, name="dp_conv5a")
    conv5 = convolution(conv5, 512, 3, strides=2, name="dp_conv5b")
    conv6 = convolution(conv5, 512, 3, strides=1, name="dp_conv6a")
    conv6 = convolution(conv6, 512, 3, strides=2, name="dp_conv6b")
    conv7 = convolution(conv6, 512, 3, strides=1, name="dp_conv7a")
    conv7 = convolution(conv7, 512, 3, strides=2, name="dp_conv7b")

    upconv7 = upconv_with_skip_connection(conv7, conv6, 512, "dp_up7")
    upconv6 = upconv_with_skip_connection(upconv7, conv5, 512, "dp_up6")
    upconv5 = upconv_with_skip_connection(upconv6, conv4, 256, "dp_up5")
    upconv4 = upconv_with_skip_connection(upconv5, conv3, 128, "dp_up4")
    disp4, disp4_up = get_disp_vgg(upconv4, int(imheight//4), int(imwidth//4), "dp_disp4")
    upconv3 = upconv_with_skip_connection(upconv4, conv2, 64, "dp_up3", disp4_up)
    disp3, disp3_up = get_disp_vgg(upconv3, int(imheight//2), int(imwidth//2), "dp_disp3")
    upconv2 = upconv_with_skip_connection(upconv3, conv1, 32, "dp_up2", disp3_up)
    disp2, disp2_up = get_disp_vgg(upconv2, imheight, imwidth, "dp_disp2")
    upconv1 = upconv_with_skip_connection(upconv2, disp2_up, 16, "dp_up1")
    disp1, disp1_up = get_disp_vgg(upconv1, imheight, imwidth, "dp_disp1")

    return [disp1, disp2, disp3, disp4]


def convolution(x, filters, kernel_size, strides, name):
    conv = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides,
                                  padding="same", activation="relu",
                                  kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                  name=name)(x)
    return conv


def upconv_with_skip_connection(bef_layer, skip_layer, out_channels, scope, bef_pred=None):
    upconv = layers.UpSampling2D(size=(2, 2), interpolation="nearest", name=scope+"_sample")(bef_layer)
    upconv = convolution(upconv, out_channels, 3, strides=1, name=scope+"_conv1")
    upconv = resize_like(upconv, skip_layer, scope)
    upconv = tf.cond(bef_pred is not None,
                     lambda: layers.Concatenate(axis=3, name=scope+"_concat")([upconv, skip_layer, bef_pred]),
                     lambda: layers.Concatenate(axis=3, name=scope+"_concat")([upconv, skip_layer])
                     )
    upconv = convolution(upconv, out_channels, 3, strides=1, name=scope+"_conv2")
    return upconv


def get_disp_vgg(x, dst_height, dst_width, scope):
    disp = layers.Conv2D(1, 3, strides=1, padding="same", activation="sigmoid", name=scope+"_conv")(x)
    disp = layers.Lambda(lambda x: DISP_SCALING_VGG * x + 0.01, name=scope+"_scale")(disp)
    disp_up = resize_image(disp, dst_height, dst_width, scope)
    return disp, disp_up


def resize_like(src, ref, scope):
    ref_height, ref_width = ref.get_shape().as_list()[1:3]
    return resize_image(src, ref_height, ref_width, scope)


def resize_image(src, dst_height, dst_width, scope):
    src_height, src_width = src.get_shape().as_list()[1:3]
    if src_height == dst_height and src_width == dst_width:
        return src
    else:
        return layers.Lambda(lambda image: tf.image.resize(
            image, size=[dst_height, dst_width], method="bilinear"), name=scope+"_resize")(src)


# ==================== build PoseNet layers ====================
def build_visual_odom_layers(stacked_image):
    channel_stack_image = layers.Lambda(lambda image: restack_on_channels(image, opts.SNIPPET_LEN),
                                        name="channel_stack")(stacked_image)
    print("[build_visual_odom_layers] channel stacked image shape=", channel_stack_image.get_shape())
    num_sources = opts.SNIPPET_LEN - 1

    conv1 = convolution(channel_stack_image, 16, 7, 2, "vo_conv1")
    conv2 = convolution(conv1, 32, 5, 2, "vo_conv2")
    conv3 = convolution(conv2, 64, 3, 2, "vo_conv3")
    conv4 = convolution(conv3, 128, 3, 2, "vo_conv4")
    conv5 = convolution(conv4, 256, 3, 2, "vo_conv5")
    conv6 = convolution(conv5, 256, 3, 2, "vo_conv6")
    conv7 = convolution(conv6, 256, 3, 2, "vo_conv7")

    poses = tf.keras.layers.Conv2D(num_sources*6, 1, strides=1, padding="same",
                                   activation=None, name="vo_conv8")(conv7)
    poses = tf.keras.layers.GlobalAveragePooling2D("channels_last", name="vo_pred")(poses)
    poses = tf.keras.layers.Reshape((num_sources, 6), name="vo_reshape")(poses)
    return poses


def restack_on_channels(vertical_stack, num_stack):
    batch, imheight, imwidth, _ = vertical_stack.get_shape().as_list()
    imheight = int(imheight // num_stack)
    # create channel for snippet sequence
    channel_stack_image = tf.reshape(vertical_stack, shape=(batch, -1, imheight, imwidth, 3))
    # move snippet dimension to 3
    channel_stack_image = tf.transpose(channel_stack_image, (0, 2, 3, 1, 4))
    # stack snippet images on channels
    channel_stack_image = tf.reshape(channel_stack_image, shape=(batch, imheight, imwidth, -1))
    return channel_stack_image


# --------------------------------------------------------------------------------
# TESTS

def test_restack_on_channels():
    print("===== start test_restack_on_channels")
    filename = op.join(opts.DATAPATH_SRC, "kitti_raw_train", "2011_09_26_0001", "000024.png")
    image = cv2.imread(filename)
    batch_image = np.expand_dims(image, 0)
    batch_image = np.tile(batch_image, (8, 1, 1, 1))
    print("batch image shape", batch_image.shape)
    batch_image_tensor = tf.constant(batch_image, dtype=tf.float32)

    channel_stack_image = restack_on_channels(batch_image_tensor, opts.SNIPPET_LEN)

    channel_stack_image = channel_stack_image.numpy()
    print("channel stack image shape", channel_stack_image.shape)
    # cv2.imshow("original image", image)
    # cv2.imshow("restack image1", channel_stack_image[0, :, :, :3])
    # cv2.imshow("restack image2", channel_stack_image[0, :, :, 3:6])
    # cv2.waitKey()
    assert (image[opts.IM_HEIGHT:opts.IM_HEIGHT*2] == channel_stack_image[0, :, :, 3:6]).all()
    print("!!! test_restack_on_channels passed")


# TODO: 바뀐 모델에서 돌아가게 수정
def test_create_models():
    model_pred, model_train = create_models()
    model_pred.summary()
    tf.keras.utils.plot_model(model_pred, to_file="model_pred.png", show_shapes=True, show_layer_names=True)
    model_train.summary()
    tf.keras.utils.plot_model(model_train, to_file="model_train.png", show_shapes=True, show_layer_names=True)
    print("!!! test_create_models passed")


def test_load_model():
    model_path = opts.DATAPATH_CKP + "/vode_model/model1.hdf5"
    print("model path", model_path)
    if op.isfile(model_path):
        model = tf.keras.models.load_model(model_path)
        model.summary()


def test():
    # test_create_models()
    # test_restack_on_channels()
    test_load_model()


if __name__ == "__main__":
    test()
