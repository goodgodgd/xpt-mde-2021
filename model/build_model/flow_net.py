import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers

import settings
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
        c1l, c2l, c3l, c4l, c5l, c6l = self.pwc_encode(target, "_l")
        c1r, c2r, c3r, c4r, c5r, c6r = self.pwc_encode(sources, "_r")

        # repeate target numsrc times -> [batch*num_src, height//scale, width//scale, channel]
        c1l, c2l, c3l, c4l, c5l, c6l = self.repeat_features((c1l, c2l, c3l, c4l, c5l, c6l), numsrc)

        corr6 = self.correlation(c6l, c6r)
        flow6, up_flow6, up_feat6 = self.predict_flow(corr6, "flow6")

        flow5, up_flow5, up_feat5 = self.upconv_flow(5, c5l, c5r, 0.625, up_flow6, up_feat6)
        flow4, up_flow4, up_feat4 = self.upconv_flow(4, c4l, c4r, 1.25,  up_flow5, up_feat5)
        flow3, up_flow3, up_feat3 = self.upconv_flow(3, c3l, c3r, 2.5,   up_flow4, up_feat4)
        flow2, flow_feat2         = self.upconv_flow(2, c2l, c2r, 5.0,   up_flow3, up_feat3, up=False)

        flow2 = self.context_network(flow_feat2, flow2)
        flows_ms = [flow2, flow3, flow4, flow5, flow6]

        # reshape back to normal bactch size
        # -> list of [batch, num_src, height//scale, width//scale, channel]
        flows_ms = self.reshape_batch_back(flows_ms)
        pwcnet = tf.keras.Model(inputs=input_tensor, outputs=flows_ms, name="PWCNet")
        return pwcnet

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

    def pwc_encode(self, x, suffix):
        c1 = self.conv2d_f(x, 16, 3, 2, name="pwc_conv1a" + suffix)
        c1 = self.conv2d_f(c1, 16, 3, 1, name="pwc_conv1b" + suffix)
        c1 = self.conv2d_f(c1, 16, 3, 1, name="pwc_conv1c" + suffix)
        c2 = self.conv2d_f(c1, 32, 3, 2, name="pwc_conv2a" + suffix)
        c2 = self.conv2d_f(c2, 32, 3, 1, name="pwc_conv2b" + suffix)
        c2 = self.conv2d_f(c2, 32, 3, 1, name="pwc_conv2c" + suffix)
        c3 = self.conv2d_f(c2, 64, 3, 2, name="pwc_conv3a" + suffix)
        c3 = self.conv2d_f(c3, 64, 3, 1, name="pwc_conv3b" + suffix)
        c3 = self.conv2d_f(c3, 64, 3, 1, name="pwc_conv3c" + suffix)
        c4 = self.conv2d_f(c3, 96, 3, 2, name="pwc_conv4a" + suffix)
        c4 = self.conv2d_f(c4, 96, 3, 1, name="pwc_conv4b" + suffix)
        c4 = self.conv2d_f(c4, 96, 3, 1, name="pwc_conv4c" + suffix)
        c5 = self.conv2d_f(c4, 128, 3, 2, name="pwc_conv5a" + suffix)
        c5 = self.conv2d_f(c5, 128, 3, 1, name="pwc_conv5b" + suffix)
        c5 = self.conv2d_f(c5, 128, 3, 1, name="pwc_conv5c" + suffix)
        c6 = self.conv2d_f(c5, 196, 3, 2, name="pwc_conv6a" + suffix)
        c6 = self.conv2d_f(c6, 196, 3, 1, name="pwc_conv6b" + suffix)
        c6 = self.conv2d_f(c6, 196, 3, 1, name="pwc_conv6c" + suffix)
        return c1, c2, c3, c4, c5, c6

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

    def upconv_flow(self, p, cp_l, cp_r, flow_scale, up_flowq, up_featq, up=True):
        """
        :param p: current layer level, q = p+1 (lower resolution level)
        :param cp_l: p-th encoded feature from left image [batch, height//2^p, width//2^p, channel_p]
        :param cp_r: p-th encoded feature from left image [batch, height//2^p, width//2^p, channel_p]
        :param flow_scale: flow scale factor for flow scale to be 1/20
        :param up_flowq: upsampled flow from q-th level [batch, height//2^p, width//2^p, 2]
        :param up_featq: upsampled flow from q-th level [batch, height//2^p, width//2^p, channel_q]
        :param up: whether to return upsample flow and feature
        :return:
        """
        cp_r_warp = layers.Lambda(lambda inputs: tfa.image.dense_image_warp(
                                  inputs[0], inputs[1]),
                                  name=f"pwc_flow{p}_warp")([cp_r, up_flowq*flow_scale])
        # cp_r_warp = tfa.image.dense_image_warp(cp_r, up_flowq * flow_scale, f"pwc_flow{p}_warp")
        corrp = self.correlation(cp_l, cp_r_warp)
        flowp_in = tf.concat([corrp, cp_l, up_flowq, up_featq], axis=-1)
        return self.predict_flow(flowp_in, f"flow{p}", up)

    def predict_flow(self, x, tag, up=True):
        c = self.conv2d_f(x, 128, name=f"pwc_{tag}_c0")
        x = tf.concat([x, c], axis=-1)
        c = self.conv2d_f(x, 128, name=f"pwc_{tag}_c1")
        x = tf.concat([x, c], axis=-1)
        c = self.conv2d_f(x, 96, name=f"pwc_{tag}_c2")
        x = tf.concat([x, c], axis=-1)
        c = self.conv2d_f(x, 64, name=f"pwc_{tag}_c3")
        x = tf.concat([x, c], axis=-1)
        c = self.conv2d_f(x, 32, name=f"pwc_{tag}_c4")
        flow = self.conv2d_f(c, 2, activation_="linear", name=f"pwc_{tag}_out")

        if up:
            up_flow = layers.Conv2DTranspose(2, kernel_size=4, strides=2, padding="same")(flow)
            up_feat = layers.Conv2DTranspose(2, kernel_size=4, strides=2, padding="same")(c)
            return flow, up_flow, up_feat
        else:
            return flow, c

    def context_network(self, x, flow):
        c = self.conv2d_f(x, 128, 3, dilation_rate_=1, name="pwc_context_1")
        c = self.conv2d_f(c, 128, 3, dilation_rate_=2, name="pwc_context_2")
        c = self.conv2d_f(c, 128, 3, dilation_rate_=4, name="pwc_context_3")
        c = self.conv2d_f(c,  96, 3, dilation_rate_=8, name="pwc_context_4")
        c = self.conv2d_f(c,  64, 3, dilation_rate_=16, name="pwc_context_5")
        c = self.conv2d_f(c,  32, 3, dilation_rate_=1, name="pwc_context_6")
        refined_flow = self.conv2d_f(c, 2, activation_="linear", name=f"pwc_context_7") + flow
        return refined_flow

    def correlation(self, cl, cr, ks=1, md=4):
        corr = tfa.layers.CorrelationCost(kernel_size=ks, max_displacement=md, stride_1=1, stride_2=1,
                                          pad=md + ks//2, data_format="channels_last")([cl, cr])
        return corr


# def convolution(x, out_channel, kernel_size=3, stride=1, dilation=1, name=None):
#     c = layers.Conv2D(out_channel, kernel_size=kernel_size, strides=stride, padding="same",
#                       dilation_rate=dilation, name=name)(x)
#     c = layers.LeakyReLU(0.1)(c)
#     return c
#
#
# def conv_flow(x, kernel_size=3, name=None):
#     c = layers.Conv2D(2, kernel_size=kernel_size, padding="same", name=name)(x)
#     return c


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
    corr = tfa.layers.CorrelationCost(kernel_size=ks, max_displacement=md, stride_1=1, stride_2=1,
                                      pad=md + ks // 2, data_format="channels_last")([cl, cr])

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


import model.build_model.model_utils as mu


def test_pwcnet():
    print("\n===== start test_pwcnet")
    total_shape = batch, snippet, height, width, channel = (8, 4, 128, 256, 10)
    input_tensor = tf.random.uniform((batch, snippet*height, width, channel), -2, 2)
    conv2d = mu.conv2d_func_factory(activation=layers.LeakyReLU(0.1))

    # EXECUTE
    pwc_net = PWCNet(total_shape, conv2d)()

    flows = pwc_net(input_tensor)
    for flow in flows:
        print("PWCNet flow shape:", flow.get_shape())
    print("!!! test_pwcnet passed")


if __name__ == "__main__":
    test_correlation()
    test_warp()
    test_conv2d_5dtensor()
    test_layer_input()
    test_reshape_tensor()
    test_pwcnet()

