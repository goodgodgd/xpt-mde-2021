import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers

import settings
import utils.util_funcs as uf
from utils.decorators import shape_check

# TODO: change function to class


class PWCNet:
    def __init__(self, total_shape, conv2d):
        self.total_shape = total_shape
        self.conv2d_f = conv2d

    def __call__(self):
        batch, snippet, height, width, channel = self.total_shape
        numsrc = snippet - 1
        input_shape = (snippet * height, width, channel)
        input_tensor = layers.Input(shape=input_shape, batch_size=batch, name="depthnet_input")
        # target: [batch, height, width, channel]
        # source: [batch*num_src, height, width, channel]
        target, sources = self.split_target_and_sources(input_tensor)

        # encode left (target) and right (source) image
        c1l, c2l, c3l, c4l, c5l, c6l = pwc_encode(target, "_l")
        c1r, c2r, c3r, c4r, c5r, c6r = pwc_encode(sources, "_r")

        # repeate target numsrc times -> [batch*num_src, height//scale, width//scale, channel]
        c1l, c2l, c3l, c4l, c5l, c6l = self.repeat_features((c1l, c2l, c3l, c4l, c5l, c6l), numsrc)

        corr6 = correlation(c6l, c6r)
        flow6, up_flow6, up_feat6 = predict_flow(corr6, "flow6")

        # flow5, up_flow5, up_feat5 = self.upconv_flow(5, c5l, c5r, up_flow6, up_feat6)
        # flow4, up_flow4, up_feat4 = self.upconv_flow(4, c4l, c4r, up_flow5, up_feat5)
        # flow3, up_flow3, up_feat3 = self.upconv_flow(3, c3l, c3r, up_flow4, up_feat4)
        # flow2, flow_feat2 =         self.upconv_flow(2, c2l, c2r, up_flow3, up_feat3, up=False)

        c5r_warp = tfa.image.dense_image_warp(c5r, up_flow6 * 0.625, "pwc_flow5_warp")
        corr5 = correlation(c5l, c5r_warp)
        flow5_in = tf.concat([corr5, c5l, up_flow6, up_feat6], axis=-1)
        flow5, up_flow5, up_feat5 = predict_flow(flow5_in, "flow5")

        c4r_warp = tfa.image.dense_image_warp(c4r, up_flow5 * 1.25, "pwc_flow4_warp")
        corr4 = correlation(c4l, c4r_warp)
        flow4_in = tf.concat([corr4, c4l, up_flow5, up_feat5], axis=-1)
        flow4, up_flow4, up_feat4 = predict_flow(flow4_in, "flow4")

        c3r_warp = tfa.image.dense_image_warp(c3r, up_flow4 * 2.5, "pwc_flow3_warp")
        corr3 = correlation(c3l, c3r_warp)
        flow3_in = tf.concat([corr3, c3l, up_flow4, up_feat4], axis=-1)
        flow3, up_flow3, up_feat3 = predict_flow(flow3_in, "flow3")

        c2r_warp = tfa.image.dense_image_warp(c2r, up_flow3 * 5.0, "pwc_flow2_warp")
        corr2 = correlation(c2l, c2r_warp)
        flow2_in = tf.concat([corr2, c2l, up_flow3, up_feat3], axis=-1)
        flow2, flow_feat2 = predict_flow(flow2_in, "flow2", up=False)

        flow2 = context_network(flow_feat2, flow2)
        flows_ms = [flow2, flow3, flow4, flow5, flow6]

        # reshape back to normal bactch size
        # -> list of [batch, num_src, height//scale, width//scale, channel]
        flows_ms = self.reshape_batch_back(flows_ms)
        return flows_ms

    def split_target_and_sources(self, input_tensor):
        """
        :param input_tensor [batch, snippet*height, width, 3]
        :return: target [batch, height, width, 3]
                 source [batch*numsrc, height, width, 3]
        """
        batch, snippet, height, width, channel = self.total_shape
        numsrc = snippet - 1
        target = input_tensor[:, numsrc*height:]
        sources = input_tensor[:, :numsrc*height]
        sources = tf.reshape(sources, (batch*numsrc, height, width, channel))
        return target, sources

    def repeat_features(self, features, numsrc):
        rep_feats = []
        for feat in features:
            batch, height, width, channel = feat.get_shape()
            # feat -> [batch, 1, height, width, channel]
            feat = tf.expand_dims(feat, 1)
            # feat -> [batch, numsrc, height, width, channel]
            feat = tf.tile(feat, (1, numsrc, 1, 1, 1))
            # feat -> [batch*numsrc, height, width, channel]
            feat = tf.reshape(feat, (batch*numsrc, height, width, channel))
            rep_feats.append(feat)
        return tuple(rep_feats)

    def reshape_batch_back(self, flows_ms):
        batch, snippet = self.total_shape[:2]
        numsrc = snippet - 1
        rsp_flows_ms = []
        for flow in flows_ms:
            _, height, width, channel = flow.get_shape()
            rsp_flow = tf.reshape(flow, (batch, numsrc, height, width, channel))
            rsp_flows_ms.append(rsp_flow)
        return rsp_flows_ms

    def upconv_flow(self, p, cp_l, cp_r, up_flowq, up_featq, up=True):
        """
        :param p: current layer level, q = p+1 (lower resolution level)
        :param cp_l: p-th encoded feature from left image [batch, height//2^p, width//2^p, channel_p]
        :param cp_r: p-th encoded feature from left image [batch, height//2^p, width//2^p, channel_p]
        :param up_flowq: upsampled flow from q-th level [batch, height//2^p, width//2^p, 2]
        :param up_featq: upsampled flow from q-th level [batch, height//2^p, width//2^p, channel_q]
        :param up: whether to return upsample flow and feature
        :return:
        """
        cp_r_warp = tfa.image.dense_image_warp(cp_r, up_flowq * 0.625, f"pwc_flow{p}_warp")
        corrp = correlation(cp_l, cp_r_warp)
        flowp_in = tf.concat([corrp, cp_l, up_flowq, up_featq], axis=-1)
        return predict_flow(flowp_in, f"flow{p}", up)

@shape_check
def pwc_encode(x, suffix):
    c1 = convolution(x,  16,  3, 2, name="pwc_conv1a" + suffix)
    c1 = convolution(c1, 16,  3, 1, name="pwc_conv1b" + suffix)
    c1 = convolution(c1, 16,  3, 1, name="pwc_conv1c" + suffix)
    c2 = convolution(c1, 32,  3, 2, name="pwc_conv2a" + suffix)
    c2 = convolution(c2, 32,  3, 1, name="pwc_conv2b" + suffix)
    c2 = convolution(c2, 32,  3, 1, name="pwc_conv2c" + suffix)
    c3 = convolution(c2, 64,  3, 2, name="pwc_conv3a" + suffix)
    c3 = convolution(c3, 64,  3, 1, name="pwc_conv3b" + suffix)
    c3 = convolution(c3, 64,  3, 1, name="pwc_conv3c" + suffix)
    c4 = convolution(c3, 96,  3, 2, name="pwc_conv4a" + suffix)
    c4 = convolution(c4, 96,  3, 1, name="pwc_conv4b" + suffix)
    c4 = convolution(c4, 96,  3, 1, name="pwc_conv4c" + suffix)
    c5 = convolution(c4, 128, 3, 2, name="pwc_conv5a" + suffix)
    c5 = convolution(c5, 128, 3, 1, name="pwc_conv5b" + suffix)
    c5 = convolution(c5, 128, 3, 1, name="pwc_conv5c" + suffix)
    c6 = convolution(c5, 196, 3, 2, name="pwc_conv6a" + suffix)
    c6 = convolution(c6, 196, 3, 1, name="pwc_conv6b" + suffix)
    c6 = convolution(c6, 196, 3, 1, name="pwc_conv6c" + suffix)

    return c1, c2, c3, c4, c5, c6


@shape_check
def predict_flow(x, tag, up=True):
    c = convolution(x, 128, 3, 1, name=f"pwc_{tag}_c0")
    x = tf.concat([x, c], axis=-1)
    c = convolution(x, 128, 3, 1, name=f"pwc_{tag}_c1")
    x = tf.concat([x, c], axis=-1)
    c = convolution(x, 96, 3, 1, name=f"pwc_{tag}_c2")
    x = tf.concat([x, c], axis=-1)
    c = convolution(x, 64, 3, 1, name=f"pwc_{tag}_c3")
    x = tf.concat([x, c], axis=-1)
    c = convolution(x, 32, 3, 1, name=f"pwc_{tag}_c4")
    flow = conv_flow(c, kernel_size=3, name=f"pwc_{tag}_out")

    if up:
        up_flow = layers.Conv2DTranspose(2, kernel_size=4, strides=2, padding="same")(flow)
        up_feat = layers.Conv2DTranspose(2, kernel_size=4, strides=2, padding="same")(c)
        return flow, up_flow, up_feat
    else:
        return flow, c


@shape_check
def context_network(x, flow):
    c = convolution(x, 128, 3, 1, dilation=1, name="pwc_context_1")
    c = convolution(c, 128, 3, 1, dilation=2, name="pwc_context_2")
    c = convolution(c, 128, 3, 1, dilation=4, name="pwc_context_3")
    c = convolution(c,  96, 3, 1, dilation=8, name="pwc_context_4")
    c = convolution(c,  64, 1, 1, dilation=16, name="pwc_context_5")
    c = convolution(c,  32, 3, 1, dilation=1, name="pwc_context_6")
    flow = conv_flow(c, kernel_size=3, name="pwc_context_7") + flow
    return flow


@shape_check
def correlation(cl, cr, ks=1, md=4):
    corr = tfa.layers.CorrelationCost(kernel_size=ks, max_displacement=md, stride_1=1, stride_2=1,
                                      pad=md + ks//2, data_format="channels_last")([cl, cr])
    return corr


def convolution(x, out_channel, kernel_size=3, stride=1, dilation=1, name=None):
    c = layers.Conv2D(out_channel, kernel_size=kernel_size, strides=stride, padding="same",
                      dilation_rate=dilation, name=name)(x)
    c = layers.LeakyReLU(0.1)(c)
    return c


def conv_flow(x, kernel_size=3, name=None):
    c = layers.Conv2D(2, kernel_size=kernel_size, padding="same", name=name)(x)
    return c


# ===== TEST FUNCTIONS

import numpy as np


def test_correlation():
    print("\n===== start test_correlation")
    batch, height, width, channel = (8, 100, 200, 10)
    cl = tf.random.uniform((batch, height, width, channel), -2, 2)
    cr = tf.random.uniform((batch, height, width, channel), -2, 2)
    print("input shape:", (batch, height, width, channel))
    ks, md = 1, 5

    # EXECUTE
    corr = correlation(cl, cr, ks, md)

    # TEST
    corr_shape = (batch, height, width, (2*md + 1)**2)
    assert corr.get_shape() == corr_shape, f"correlation shape: {corr.get_shape()} != {corr_shape}"
    print("correlation shape:", corr.get_shape())

    # manually compute correlation at (md+v, md+u) but NOT same with corr
    u, v = 1, 1
    cr_shift = tf.pad(cr[:, v:, u:, :], [[0, 0], [v, 0], [u, 0], [0, 0]])
    corr_man = cl * cr_shift
    corr_man = layers.AveragePooling2D(pool_size=(ks, ks), strides=1, padding="same")(corr_man)
    corr_man = tf.reduce_mean(corr_man, axis=-1)
    print("corr_man shape:", corr_man.get_shape())

    print("!!! test_correlation passed")


def test_warp():
    print("\n===== start test_warp")
    batch, height, width, channel = (8, 100, 200, 10)
    im = tf.random.uniform((batch, height, width, channel), -2, 2)
    dy, dx = 1.5, 0.5
    flow = tf.stack([tf.ones((batch, height, width)) * dy, tf.ones((batch, height, width)) * dx], axis=-1)

    # EXECUTE
    warp_tfa = tfa.image.dense_image_warp(im, flow)

    # flow is applied in a negative way
    warp_man = (im[:, 9:19, 10:20, :] + im[:, 8:18, 10:20, :] + im[:, 9:19, 9:19, :]
                + im[:, 8:18, 9:19, :]) / 4.
    print("warp_tfa:", warp_tfa[1, 10:15, 10:15, 1].numpy())
    print("warp_man:", warp_man[1, 0:5, 0:5, 1].numpy())
    assert np.isclose(warp_tfa[1, 10:15, 10:15, 1].numpy(), warp_man[1, 0:5, 0:5, 1].numpy()).all()
    print("!!! test_warp passed")


def test_conv2d_5dtensor():
    print("\n===== start test_conv2d_5dtensor")
    input_shape = (8, 4, 100, 200, 10)
    input_tensor = tf.random.uniform(input_shape, -2, 2)
    conv_layer = tf.keras.layers.Conv2D(100, 3, 1, "same")
    try:
        output_tensor = conv_layer(input_tensor)
        print(output_tensor.get_shape())
        print("!!! test_pwcnet passed")
    except ValueError as ve:
        print("ERROR!!")
        print("[test_conv2d_5dtensor]", ve)


def test_layer_input():
    print("\n===== start test_layer_input")
    batch, numsrc, height, width, channel = (8, 4, 100, 200, 10)
    input_tensor = layers.Input(shape=(height, width, channel), batch_size=batch*numsrc, name="input_tensor")
    print("input tensor shape:", input_tensor.get_shape())
    assert input_tensor.get_shape() == (batch*numsrc, height, width, channel)
    print("!!! test_layer_input passed")


def test_reshape_tensor():
    print("\n===== start test_reshape_tensor")
    batch, numsrc, height, width, channel = (8, 4, 100, 200, 10)
    src_tensor = tf.random.uniform((batch, numsrc*height, width, channel), -2, 2)
    dst_tensor = tf.reshape(src_tensor, (batch*numsrc, height, width, channel))
    batidx = 1
    srcidx = 1
    print("src data:\n", src_tensor[batidx, height*srcidx + 5:height*srcidx + 10, 5:10, 3].numpy())
    print("src data:\n", dst_tensor[numsrc*batidx + srcidx, 5:10, 5:10, 3].numpy())
    assert np.isclose(src_tensor[batidx, height*srcidx:height*(srcidx+1)].numpy(),
                      dst_tensor[numsrc*batidx + srcidx, :].numpy()).all()
    print("!!! test_reshape_tensor passed")


def test_pwcnet():
    print("\n===== start test_pwcnet")
    batch, height, width, channel = (8, 128, 256, 10)
    xl = tf.random.uniform((batch, height, width, channel), -2, 2)
    xr = tf.random.uniform((batch, height, width, channel), -2, 2)

    # EXECUTE
    flows = pwcnet(xl, xr)

    flow2 = flows[0]
    assert flow2.get_shape() == (batch, height//4, width//4, 2)
    print("!!! test_pwcnet passed")


if __name__ == "__main__":
    test_correlation()
    test_warp()
    test_conv2d_5dtensor()
    test_layer_input()
    test_reshape_tensor()

