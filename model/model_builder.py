import os.path as op
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import settings
from config import opts
from model.loss_and_metric import synthesize_view_multi_scale


def create_models(image_shape, intrin_shape, batch_size):
    model_pred = create_pred_model(image_shape, intrin_shape, batch_size)
    # model_train = create_train_model(model_pred)
    # return model_pred, model_train


def create_pred_model(image_shape, intrin_shape, batch_size):
    # prepare input tensors
    stacked_image = layers.Input(shape=image_shape, batch_size=batch_size, name="target")
    intrinsic = layers.Input(shape=intrin_shape, batch_size=batch_size, name="intrinsic")
    model_input = {"image": stacked_image, "intrinsic": intrinsic}

    # build layers of posenet and depthnet and make model for prediction
    pred_depths_ms = build_depth_estim_layers(stacked_image)
    pred_poses = build_visual_odom_layers(stacked_image)
    predictions = {**pred_depths_ms, "pose": pred_poses}
    model_pred = tf.keras.Model(model_input, predictions)
    model_pred.compile(optimizer="adam", loss="mean_absolute_error")
    return model_pred


def create_train_model(model_pred):
    # calculate loss and make model for training
    model_inputs = model_pred.input
    stacked_image = model_inputs["image"]
    intrinsic = model_inputs["intrinsic"]
    predictions = model_pred.output
    depth_ms = [value for key, value in predictions.items() if key.startsWith("depth")]
    pose = predictions["pose"]
    synthesized_targets_ms = synthesize_view_multi_scale(stacked_image, intrinsic, depth_ms, pose)
    # loss = synthesized_targets_ms
    # # loss += calc_photometric_loss(synthesized_targets_ms, target_input)
    # model_train = tf.keras.Model(model_input, loss)
    # model_train.compile(optimizer="adam", loss="mean_absolute_error")
    # return model_train
    pass


# ==================== build DepthNet layers ====================
DISP_SCALING_VGG = 10


def build_depth_estim_layers(stacked_image):
    batch, imheight, imwidth, imchannel = stacked_image.get_shape().as_list()
    imheight = int(imheight // opts.SNIPPET_LEN)
    target_image = layers.Lambda(lambda image: tf.slice(image,
                                 (0, imheight*(opts.SNIPPET_LEN-1), 0, 0),
                                 (-1, imheight, -1, -1)),
                                 name="extract_target")(stacked_image)
    print("[build_depth_estim_layers] target image shape=", target_image.get_shape())

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
    pred4, pred4_up = get_disp_vgg(upconv4, int(imheight//4), int(imwidth//4), "dp_pred4")
    upconv3 = upconv_with_skip_connection(upconv4, conv2, 64, "dp_up3", pred4_up)
    pred3, pred3_up = get_disp_vgg(upconv3, int(imheight//2), int(imwidth//2), "dp_pred3")
    upconv2 = upconv_with_skip_connection(upconv3, conv1, 32, "dp_up2", pred3_up)
    pred2, pred2_up = get_disp_vgg(upconv2, imheight, imwidth, "dp_pred2")
    upconv1 = upconv_with_skip_connection(upconv2, pred2_up, 16, "dp_up1")
    pred1 = get_disp_vgg(upconv1, imheight, imwidth, "dp_pred1")

    return {"depth1": pred1, "depth2": pred2, "depth3": pred3, "depth4": pred4}


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
                                   activation=None, name="vo_pred")(conv7)
    poses = tf.keras.backend.mean(poses, axis=(1, 2))
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


# ==================== tests ====================
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
    print("test_restack_on_channels passed")


def test_create_pred_model():
    image_shape = (opts.IM_HEIGHT*opts.SNIPPET_LEN, opts.IM_WIDTH, 3)
    intrin_shape = (3, 3)
    model_pred = create_pred_model(image_shape, intrin_shape, batch_size=8)
    model_pred.summary()
    tf.keras.utils.plot_model(model_pred, to_file="model.png", show_shapes=True, show_layer_names=True)


# ==================== tests ====================
def test():
    test_create_pred_model()
    test_restack_on_channels()


if __name__ == "__main__":
    test()