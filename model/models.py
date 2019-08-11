import tensorflow as tf
from tensorflow.keras import layers


def create_model(target_shape, source_shape, batch_size):
    target_input = layers.Input(shape=target_shape, batch_size=batch_size, name="target_input")
    source_input = layers.Input(shape=source_shape, batch_size=batch_size, name="source_input")
    stacked_input = layers.Concatenate(axis=3, name="stacked_input")([target_input, source_input])
    inputs = {"target": target_input, "source": source_input}

    model_out = build_depth_estim_layers(target_input)
    odom_out = build_visual_odom_layers(stacked_input)
    model_out["pose"] = odom_out

    model = tf.keras.Model(inputs, model_out)
    model.compile(optimizer="adam", loss="mean_absolute_error")
    return model


# ==================== build DepthNet layers ====================
DISP_SCALING_VGG = 10


def build_depth_estim_layers(image_input):
    batch, imheight, imwidth, _ = image_input.get_shape().as_list()

    conv1 = convolution(image_input, 32, 7, strides=1, name="dp_conv1a")
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
    pred4 = get_disp_vgg(upconv4, int(imheight//4), int(imwidth//4), "dp_pred4")
    upconv3 = upconv_with_skip_connection(upconv4, conv2, 64, "dp_up3", pred4)
    pred3 = get_disp_vgg(upconv3, int(imheight//2), int(imwidth//2), "dp_pred3")
    upconv2 = upconv_with_skip_connection(upconv3, conv1, 32, "dp_up2", pred3)
    pred2 = get_disp_vgg(upconv2, imheight, imwidth, "dp_pred2")
    upconv1 = upconv_with_skip_connection(upconv2, pred2, 16, "dp_up1")
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
    disp = resize_image(disp, dst_height, dst_width, scope)
    return disp


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
def build_visual_odom_layers(stacked_input):
    num_sources = int(stacked_input.get_shape().as_list()[3] // 3 - 1)

    conv1 = convolution(stacked_input, 16, 7, 2, "vo_conv1")
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


# ==================== executable ====================
def test():
    model = create_model(target_shape=(128, 416, 3), source_shape=(128, 416, 12), batch_size=8)
    model.summary()
    tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)


if __name__ == "__main__":
    test()
