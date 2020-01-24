import os.path as op
import tensorflow as tf
from tensorflow.keras import layers

import settings
from config import opts
import utils.util_funcs as uf


def create_model():
    # input tensor
    image_shape = (opts.IM_HEIGHT * opts.SNIPPET_LEN, opts.IM_WIDTH, 3)
    stacked_image = layers.Input(shape=image_shape, batch_size=opts.BATCH_SIZE, name="image")
    source_image, target_image = layers.Lambda(lambda image: uf.split_into_source_and_target(image),
                                               name="split_stacked_image")(stacked_image)
    # build a network that outputs depth and pose
    pred_disps_ms = build_depth_estim_layers(target_image)
    pred_poses = build_visual_odom_layers(stacked_image)
    # create model
    model_input = {"image": stacked_image}
    predictions = {"disp_ms": pred_disps_ms, "pose": pred_poses}
    model = tf.keras.Model(model_input, predictions)
    return model


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
                                  kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.025),
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

from tfrecords.tfrecord_reader import TfrecordGenerator


def test_restack_on_channels():
    print("===== start test_restack_on_channels")
    batch_size = 4
    dataset = TfrecordGenerator(op.join(opts.DATAPATH_TFR, "kitti_raw_test"), batch_size=batch_size).get_generator()
    itdataset = iter(dataset)
    features = next(itdataset)
    vertical_stack_image = features['image']

    channel_stack_image = restack_on_channels(vertical_stack_image, opts.SNIPPET_LEN)

    vertical_stack_image = tf.image.convert_image_dtype((vertical_stack_image + 1.) / 2., dtype=tf.uint8).numpy()
    channel_stack_image = tf.image.convert_image_dtype((channel_stack_image + 1.) / 2., dtype=tf.uint8).numpy()
    print("channel stacked image shape:", channel_stack_image.shape)
    # 위아래로 쌓인 이미지를 채널로 잘 쌓았는지 shape 확인
    assert (channel_stack_image.shape == (batch_size, opts.IM_HEIGHT, opts.IM_WIDTH, opts.SNIPPET_LEN*3))
    # 이미지가 맞게 배치됐는지 확인: snippet에서 두번째 이미지를 비교
    assert (vertical_stack_image[1, opts.IM_HEIGHT:opts.IM_HEIGHT*2, :, :] == channel_stack_image[1, :, :, 3:6]).all()
    print("!!! test [restack_on_channels] passed")
    # cv2.imshow("original image full", vertical_stack_image[0])
    # cv2.imshow("restack image0", channel_stack_image[0, :, :, :3])
    # cv2.imshow("restack image1", channel_stack_image[0, :, :, 3:6])
    # cv2.waitKey()


def test_create_model():
    # load model and plot the network structure to a png file
    model = create_model()
    model.summary()
    tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)
    print("!!! test_create_models passed")


def test():
    test_restack_on_channels()
    # test_create_model()


if __name__ == "__main__":
    test()
