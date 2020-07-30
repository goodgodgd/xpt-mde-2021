import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
from model.synthesize.bilinear_interp import FlowBilinearInterpolation

import settings


class PWCNet:
    def __init__(self, total_shape, conv2d):
        self.total_shape = total_shape
        self.conv2d_f = conv2d
        # maximum pixel movement in optical flow
        self.max_displacement = 128
        print("[FlowNet] convolution default options:", vars(conv2d))

    def __call__(self):
        batch, snippet, height, width, channel = self.total_shape
        numsrc = snippet - 1
        input_shape = (snippet, height, width, channel)
        input_tensor = layers.Input(shape=input_shape, batch_size=batch, name="flownet_input")
        # target: [batch, height, width, channel]
        # source: [batch*numsrc, height, width, channel]
        target, sources = layers.Lambda(lambda x: self.split_target_and_sources(x),
                                        name="input_split")(input_tensor)

        # encode left (target) and right (source) image
        c1l, c2l, c3l, c4l, c5l, c6l = PWCEncoder(self.conv2d_f, "left_encode")(target)
        c1r, c2r, c3r, c4r, c5r, c6r = PWCEncoder(self.conv2d_f, "right_encode")(sources)

        # repeate target numsrc times -> [batch*numsrc, height/scale, width/scale, channel]
        c1l, c2l, c3l, c4l, c5l, c6l = RepeatFeatures(numsrc, "repeat")((c1l, c2l, c3l, c4l, c5l, c6l))

        corr6 = correlation(c6l, c6r, self.max_displacement, 6, "flow6_corr")
        feat6, flow6 = FlowPredictor("flow_predict6")(corr6)
        up_flow6, up_feat6 = self.conv_transpose(feat6, flow6, "pwc_upconv6")

        feat5, flow5 = self.decode_flow(5, c5l, c5r, 0.625, up_flow6, up_feat6)
        up_flow5, up_feat5 = self.conv_transpose(feat5, flow5, "pwc_upconv5")

        feat4, flow4 = self.decode_flow(4, c4l, c4r, 1.25,  up_flow5, up_feat5)
        up_flow4, up_feat4 = self.conv_transpose(feat4, flow4, "pwc_upconv4")

        feat3, flow3 = self.decode_flow(3, c3l, c3r, 2.5,   up_flow4, up_feat4)
        up_flow3, up_feat3 = self.conv_transpose(feat3, flow3, "pwc_upconv3")

        feat2, flow2 = self.decode_flow(2, c2l, c2r, 5.0,   up_flow3, up_feat3)
        flow2 = ContextNetwork("pwc_context")([feat2, flow2])

        flow_ms = [flow2, flow3, flow4, flow5]

        # reshape back to normal bactch size
        # -> list of [batch, numsrc, height/scale, width/scale, 2]
        flow_ms = layers.Lambda(lambda inputs: self.reshape_batch_back(inputs))(flow_ms)
        pwcnet = tf.keras.Model(inputs=input_tensor, outputs={"flow_ms": flow_ms}, name="PWCNet")
        return pwcnet

    def split_target_and_sources(self, input_tensor):
        """
        :param input_tensor [batch, snippet*height, width, 3]
        :return: target [batch, height, width, 3]
                 source [batch*numsrc, height, width, 3]
        """
        batch, snippet, height, width, channel = self.total_shape
        numsrc = snippet - 1
        target = input_tensor[:, -1]
        sources = input_tensor[:, :-1]
        sources = tf.reshape(sources, (batch*numsrc, height, width, channel))
        return target, sources

    def reshape_batch_back(self, flows_ms):
        batch, snippet = self.total_shape[:2]
        numsrc = snippet - 1
        rsp_flows_ms = []
        for flow in flows_ms:
            _, height, width, channel = flow.get_shape()
            rsp_flow = tf.reshape(flow, (batch, numsrc, height, width, channel))
            rsp_flows_ms.append(rsp_flow)
        return rsp_flows_ms

    def decode_flow(self, p, cp_l, cp_r, flow_scale, up_flowq, up_featq):
        """
        :param p: current layer level, q = p+1, feature resolution is (H/2^p, W/2^p)
        :param cp_l: p-th encoded feature from left image [batch*numsrc, height/2^p, width/2^p, channel_p]
        :param cp_r: p-th encoded feature from left image [batch*numsrc, height/2^p, width/2^p, channel_p]
        :param flow_scale: flow scale factor for flow scale to be 1/20
        :param up_flowq: upsampled flow from q-th level [batch*numsrc, height/2^p, width/2^p, 2]
        :param up_featq: upsampled flow from q-th level [batch*numsrc, height/2^p, width/2^p, channel_q]
        :return:
        """
        corrp = FlowFeature(self.max_displacement, p, flow_scale, f"pwc_decode{p}")([cp_l, cp_r, up_flowq])
        feat, flow = FlowPredictor(f"flow_predict{p}")([corrp, cp_l, up_flowq, up_featq])
        return feat, flow

    def conv_transpose(self, feat, flow, prefix):
        up_feat = layers.Conv2DTranspose(2, kernel_size=4, strides=2, padding="same",
                                         name=prefix + "ct2")(feat)
        up_flow = layers.Conv2DTranspose(2, kernel_size=4, strides=2, padding="same",
                                         name=prefix + "ct1")(flow)
        return up_feat, up_flow


class PWCEncoder(layers.Layer):
    def __init__(self, conv2d, name):
        super().__init__(name=name)
        self.conv2d_f = conv2d

    def call(self, x, **kwargs):
        c1 = self.conv2d_f(x, 16, 3, 2)
        c1 = self.conv2d_f(c1, 16, 3, 1)
        c1 = self.conv2d_f(c1, 16, 3, 1)
        c2 = self.conv2d_f(c1, 32, 3, 2)
        c2 = self.conv2d_f(c2, 32, 3, 1)
        c2 = self.conv2d_f(c2, 32, 3, 1)
        c3 = self.conv2d_f(c2, 64, 3, 2)
        c3 = self.conv2d_f(c3, 64, 3, 1)
        c3 = self.conv2d_f(c3, 64, 3, 1)
        c4 = self.conv2d_f(c3, 96, 3, 2)
        c4 = self.conv2d_f(c4, 96, 3, 1)
        c4 = self.conv2d_f(c4, 96, 3, 1)
        c5 = self.conv2d_f(c4, 128, 3, 2)
        c5 = self.conv2d_f(c5, 128, 3, 1)
        c5 = self.conv2d_f(c5, 128, 3, 1)
        c6 = self.conv2d_f(c5, 196, 3, 2)
        c6 = self.conv2d_f(c6, 196, 3, 1)
        c6 = self.conv2d_f(c6, 196, 3, 1)
        return c1, c2, c3, c4, c5, c6


class RepeatFeatures(layers.Layer):
    def __init__(self, numsrc, name):
        super().__init__(name=name)
        self.numsrc = numsrc

    def call(self, features, **kwargs):
        rep_feats = []
        for feat in features:
            batch, height, width, channel = feat.get_shape()
            # feat -> [batch, 1, height, width, channel]
            feat = tf.expand_dims(feat, 1)
            # feat -> [batch, numsrc, height, width, channel]
            feat = tf.tile(feat, (1, self.numsrc, 1, 1, 1))
            # feat -> [batch*numsrc, height, width, channel]
            feat = tf.reshape(feat, (batch*self.numsrc, height, width, channel))
            rep_feats.append(feat)
        return tuple(rep_feats)


class FlowFeature(layers.Layer):
    def __init__(self, md, p, flow_scale, name):
        """
        :param md: max displacement of flow
        :param p: current layer level, q = p+1, feature resolution is (H/2^p, W/2^p)
        :param flow_scale: flow scale factor for flow scale to be 1/20
        """
        super().__init__(name=name)
        self.max_displacement = md
        self.scale_index = p
        self.flow_scale = flow_scale

    def call(self, inputs, **kwargs):
        cp_l, cp_r, up_flowq = inputs
        cp_r_warp = layers.Lambda(lambda inps: tfa.image.dense_image_warp(inps[0], inps[1]*self.flow_scale),
                                  )([cp_r, up_flowq])
        corrp = correlation(cp_l, cp_r_warp, self.md, self.scale_index)
        return corrp


class FlowPredictor(layers.Layer):
    def __init__(self, name):
        super().__init__(name)

    def call(self, input_tensors, **kwargs):
        x = tf.concat(input_tensors, axis=-1)
        c = self.conv2d_f(x, 128)
        x = tf.concat([x, c], axis=-1)
        c = self.conv2d_f(x, 128)
        x = tf.concat([x, c], axis=-1)
        c = self.conv2d_f(x, 96)
        x = tf.concat([x, c], axis=-1)
        c = self.conv2d_f(x, 64)
        x = tf.concat([x, c], axis=-1)
        c = self.conv2d_f(x, 32)
        flow = self.conv2d_f(c, 2, activation="linear")
        return c, flow


class ContextNetwork(layers.Layer):
    def __init__(self, name):
        super().__init__(name=name)

    def call(self, inputs, **kwargs):
        x, flow = inputs
        c = self.conv2d_f(x, 128, 3, dilation_rate=1)
        c = self.conv2d_f(c, 128, 3, dilation_rate=2)
        c = self.conv2d_f(c, 128, 3, dilation_rate=4)
        c = self.conv2d_f(c,  96, 3, dilation_rate=8)
        c = self.conv2d_f(c,  64, 3, dilation_rate=16)
        c = self.conv2d_f(c,  32, 3, dilation_rate=1)
        refined_flow = self.conv2d_f(c, 2, activation="linear") + flow
        return refined_flow


def correlation(cl, cr, max_displacement, p, name=""):
    """
    :param cl: left convolutional features [batch, height/2^p, width/2^p, channels]
    :param cr: right convolutional features [batch, height/2^p, width/2^p, channels]
    :param max_displacement: max displacement
    :param p: resolution level
    :param name:
    :return: correlation volumn [batch, height/2^p, width/2^p, (2*md+1)^2]
    """
    md = max_displacement // 2**p
    stride_2 = max(md//4, 1)
    corr = tfa.layers.CorrelationCost(kernel_size=1, max_displacement=md,
                                      stride_1=1, stride_2=stride_2,
                                      pad=md, data_format="channels_last", name=name
                                      )([cl, cr])
    print(f"[CorrelationCost] max_displacement={md}, stride_2={stride_2}, corr shape={corr.shape}")
    return corr


# ===== TEST FUNCTIONS

import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=100)


def test_correlation():
    print("\n===== start test_correlation")
    batch, height, width, channel = (8, 128, 384, 10)
    cl = tf.random.uniform((batch, height, width, channel), -2, 2)
    cr = tf.random.uniform((batch, height, width, channel), -2, 2)
    print("input shape:", (batch, height, width, channel))
    max_displacement = 128
    resolution_level = range(2, 7)
    ks = 1

    for p in resolution_level:
        md = max_displacement // 2**p
        stride_2 = max(md//4, 1)
        corr = tfa.layers.CorrelationCost(kernel_size=ks, max_displacement=md, stride_1=1, stride_2=stride_2,
                                          pad=md + ks//2, data_format="channels_last")([cl, cr])
        print(f"Level={p}, md={md}, stride_2={stride_2}, corr shape={corr.shape}")

    print("correlation channels are kept constant")
    print("!!! test_correlation passed")


def test_warp_simple():
    print("\n===== start test_warp_simple")
    batch, height, width, channel = (8, 100, 200, 10)
    im = tf.random.uniform((batch, height, width, channel), -2, 2)
    dy, dx = 3.5, 1.5
    dyd, dyu, dxd, dxu = int(np.floor(dy)), int(np.ceil(dy)), int(np.floor(dx)), int(np.ceil(dx))
    print("dy up, dy down, dx up, dx down:", dyd, dyu, dxd, dxu)
    # dense_image_warp needs warp: [batch, height, width, 2]
    warp_vu = tf.stack([tf.ones((batch, height, width)) * dy, tf.ones((batch, height, width)) * dx], axis=-1)
    warp_uv = tf.stack([tf.ones((batch, height, width)) * dx, tf.ones((batch, height, width)) * dy], axis=-1)

    # EXECUTE
    warp_tfa = tfa.image.dense_image_warp(im, warp_vu)
    warp_ian = FlowBilinearInterpolation()(im, warp_uv)

    # sample image and warped images
    im_np = im[1, :, :, 1].numpy()
    warp_tfa_np = warp_tfa[1, :, :, 1].numpy()
    warp_ian_np = warp_ian[1, :, :, 1].numpy()
    # create manually interpolated image
    temp = (im_np[:-dyu, :-dxu] + im_np[1:-dyd, :-dxu] + im_np[:-dyu, 1:-dxd] + im_np[1:-dyd, 1:-dxd])/4.
    interp_manual = np.zeros((height, width))
    interp_manual[dyu:, dxu:] += temp

    # TEST
    assert np.isclose(warp_tfa_np[dyu:, dxu:], warp_ian_np[dyu:, dxu:]).all()
    assert np.isclose(interp_manual[dyu:, dxu:], warp_ian_np[dyu:, dxu:]).all()

    print("src image corner:\n", im_np[:8, :8])
    print(f"image corner warped by dense_image_warp: {warp_tfa.get_shape()}\n", warp_tfa_np[:8, :8])
    print(f"image corner warped by FlowBilinearInterpolation: {warp_ian.get_shape()}\n", warp_ian_np[:8, :8])
    print(f"image corner manually interpolated: {interp_manual.shape}\n", interp_manual[:8, :8])
    print("!!! test_warp_simple passed")
    return


def test_warp_multiple():
    print("\n===== start test_warp_simple")

    for k in range(1, 10):
        batch, height, width, channel = (8, 100*5, 200, 10)
        im = tf.random.uniform((batch, height, width, channel), -2, 2)
        dy, dx = 1.5, 0.5
        flow = tf.stack([tf.ones((batch, height, width)) * dy, tf.ones((batch, height, width)) * dx], axis=-1)

        # EXECUTE
        warp_tfa = tfa.image.dense_image_warp(im, flow)
        print("dense_image_warp without name", warp_tfa.get_shape())

    # TODO: WARNING!! the below loop results in warnings like
    #   "WARNING:tensorflow:5 out of the last 13 calls to <function dense_image_warp at 0x7f1c2e87e7a0>
    #   triggered tf.function retracing. ~~~"
    #   It seems like a bug and it happens only in the eager execution mode
    for k in range(1, 10):
        batch, height, width, channel = (8, 100*5, 200, 10)
        im = tf.random.uniform((batch, height, width, channel), -2, 2)
        dy, dx = 1.5, 0.5
        flow = tf.stack([tf.ones((batch, height, width)) * dy, tf.ones((batch, height, width)) * dx], axis=-1)

        # EXECUTE
        warp_tfa = tfa.image.dense_image_warp(im, flow, name=f"warp{k}")
        print("dense_image_warp with name", warp_tfa.get_shape())

    print("!!! test_warp_simple passed")


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


import model.model_util.layer_ops as lo


def test_lambda_layer():
    print("\n===== start test_lambda_layer")
    batch, height, width, channel = (8, 100, 200, 10)
    x = tf.random.uniform((batch, height, width, channel), -2, 2)
    conv2d = lo.CustomConv2D(activation=layers.LeakyReLU(0.1))
    y = convnet(conv2d, x)
    print("normally build convnet, output shape:", y.get_shape())

    try:
        y = layers.Lambda(lambda inputs: convnet(conv2d, inputs), name=f"convnet")(x)
        print("!!! test_lambda_layer passed")
    except ValueError as ve:
        print("!!! Exception raised in test_lambda_layer:", ve)


def convnet(conv_op, x):
    c = conv_op(x, 3)
    c = conv_op(c, 5)
    c = conv_op(c, 1, strides=2)
    return c


def test_pwcnet():
    print("\n===== start test_pwcnet")
    total_shape = batch, snippet, height, width, channel = (8, 4, 128, 256, 10)
    input_tensor = tf.random.uniform(total_shape, -2, 2)
    conv_layer = lo.CustomConv2D(activation=tf.keras.layers.LeakyReLU(0.1),
                                 kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.025),
                                 kernel_regularizer=tf.keras.regularizers.l2(0.0004)
                                 )

    # EXECUTE
    pwc_net = PWCNet(total_shape, conv_layer)()
    # pwc_net.summary()

    flow_ms = run_net(pwc_net, input_tensor)
    flow_ms = flow_ms["flow_ms"]
    for flow in flow_ms:
        print("PWCNet flow shape:", flow.get_shape())
    assert flow_ms[0].get_shape() == (batch, snippet - 1, height // 4, width // 4, 2)
    assert flow_ms[1].get_shape() == (batch, snippet - 1, height // 8, width // 8, 2)
    print("!!! test_pwcnet passed")


# @tf.function
def run_net(net, input_tensor):
    return net(input_tensor)


if __name__ == "__main__":
    test_correlation()
    test_warp_simple()
    test_warp_multiple()
    test_conv2d_5dtensor()
    test_layer_input()
    test_reshape_tensor()
    test_lambda_layer()
    test_pwcnet()

