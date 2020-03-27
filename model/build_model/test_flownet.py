import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

import settings
import model.build_model.model_utils as mu
from model.synthesize.bilinear_interp import BilinearInterpolation


def pwcnet(xl, xr):
    # encode left and right image
    c1l, c2l, c3l, c4l, c5l, c6l = pwc_encode(xl, "_l")
    c1r, c2r, c3r, c4r, c5r, c6r = pwc_encode(xr, "_r")

    corr6 = correlation(c6l, c6r)
    flow6, up_flow6, up_feat6 = predict_flow(corr6, "flow6")

    warp5 = warp(c5r, up_flow6 * 0.625)
    corr5 = correlation(c5l, warp5)
    flow5_in = tf.concat([corr5, c5l, up_flow6, up_feat6], axis=-1)
    flow5, up_flow5, up_feat5 = predict_flow(flow5_in, "flow5")

    warp4 = warp(c4r, up_flow5 * 1.25)
    corr4 = correlation(c4l, warp4)
    flow4_in = tf.concat([corr4, c4l, up_flow5, up_feat5], axis=-1)
    flow4, up_flow4, up_feat4 = predict_flow(flow4_in, "flow4")

    warp3 = warp(c3r, up_flow4 * 2.5)
    corr3 = correlation(c3l, warp3)
    flow3_in = tf.concat([corr3, c3l, up_flow4, up_feat4], axis=-1)
    flow3, up_flow3, up_feat3 = predict_flow(flow3_in, "flow3")

    warp2 = warp(c2r, up_flow3 * 5.0)
    corr2 = correlation(c2l, warp2)
    flow2_in = tf.concat([corr2, c2l, up_flow3, up_feat3], axis=-1)
    flow2, flow_feat2 = predict_flow(flow2_in, "flow2", up=False)

    flow2 = context_network(flow_feat2, flow2)

    return flow2, flow3, flow4, flow5, flow6


def pwc_encode(x, suffix):
    c1 = mu.convolution(x,  16,  3, 2, "pwc_conv1a" + suffix)
    c1 = mu.convolution(c1, 16,  3, 1, "pwc_conv1b" + suffix)
    c1 = mu.convolution(c1, 16,  3, 1, "pwc_conv1c" + suffix)
    c2 = mu.convolution(c1, 32,  3, 2, "pwc_conv2a" + suffix)
    c2 = mu.convolution(c2, 32,  3, 1, "pwc_conv2b" + suffix)
    c2 = mu.convolution(c2, 32,  3, 1, "pwc_conv2c" + suffix)
    c3 = mu.convolution(c2, 64,  3, 2, "pwc_conv3a" + suffix)
    c3 = mu.convolution(c3, 64,  3, 1, "pwc_conv3b" + suffix)
    c3 = mu.convolution(c3, 64,  3, 1, "pwc_conv3c" + suffix)
    c4 = mu.convolution(c3, 96,  3, 2, "pwc_conv4a" + suffix)
    c4 = mu.convolution(c4, 96,  3, 1, "pwc_conv4b" + suffix)
    c4 = mu.convolution(c4, 96,  3, 1, "pwc_conv4c" + suffix)
    c5 = mu.convolution(c4, 128, 3, 2, "pwc_conv5a" + suffix)
    c5 = mu.convolution(c5, 128, 3, 1, "pwc_conv5b" + suffix)
    c5 = mu.convolution(c5, 128, 3, 1, "pwc_conv5c" + suffix)
    c6 = mu.convolution(c5, 196, 3, 2, "pwc_conv6a" + suffix)
    c6 = mu.convolution(c6, 196, 3, 1, "pwc_conv6b" + suffix)
    c6 = mu.convolution(c6, 196, 3, 1, "pwc_conv6c" + suffix)

    return c1, c2, c3, c4, c5, c6


def warp_feature(x, flow):
    """
    :param x: [batch, height, width, channel]
    :param flow: (du, dv) [batch, height, width, 2]
    :return:
    """
    batch, height, width, chann = x.get_shape()
    v = np.linspace(0, height, height).astype(np.float32)
    u = np.linspace(0, width,  width).astype(np.float32)
    ugrid, vgrid = tf.meshgrid(u, v)
    vu = tf.stack([vgrid, ugrid], axis=-2)
    pixel = vu + flow
    return x


def correlation(cl, cr, md=4):
    """
    :param cl: convolutional feature of left image, [batch, height, width, channel]
    :param cr: convolutional feature of right image, [batch, height, width, channel]
    :param md: maximum displacement
    :return:
    """
    corr = []
    for u in range(-md, md + 1):
        for v in range(-md, md + 1):
            # shift left feature and fill zeros to empty elements
            if (v >= 0) and (u >= 0):
                cr_crop = cr[:, v:, u:, :]
                cr_pad = tf.pad(cr_crop, [[0, 0], [v, 0], [u, 0], [0, 0]])
            elif (v < 0) and (u >= 0):
                cr_crop = cr[:, :v, u:, :]
                cr_pad = tf.pad(cr_crop, [[0, 0], [0, v], [u, 0], [0, 0]])
            elif (v >= 0) and (u < 0):
                cr_crop = cr[:, v:, :u, :]
                cr_pad = tf.pad(cr_crop, [[0, 0], [v, 0], [0, u], [0, 0]])
            else:
                cr_crop = cr[:, :v, :u, :]
                cr_pad = tf.pad(cr_crop, [[0, 0], [0, v], [0, u], [0, 0]])

            corr_ch = layers.AveragePooling2D(pool_size=(3, 3), strides=1, padding="same")\
                (cl * cr_pad)
            # [batch, height, width, 1]
            corr_ch = tf.reduce_mean(corr_ch, axis=-1)
            corr.append(corr_ch)

    corr = tf.stack(corr, axis=-1)
    corr = layers.LeakyReLU(0.1)(corr)
    return corr


def predict_flow(x, tag, up=True):
    c = mu.convolution(x, 128, 3, 1, f"pwc_{tag}_c0")
    x = tf.concat([x, c])
    c = mu.convolution(x, 128, 3, 1, f"pwc_{tag}_c1")
    x = tf.concat([x, c])
    c = mu.convolution(x, 96, 3, 1, f"pwc_{tag}_c2")
    x = tf.concat([x, c])
    c = mu.convolution(x, 64, 3, 1, f"pwc_{tag}_c3")
    x = tf.concat([x, c])
    c = mu.convolution(x, 32, 3, 1, f"pwc_{tag}_c4")
    flow = mu.convolution(c, 2, 3, 1, f"pwc_{tag}_out")

    if up:
        up_flow = layers.Conv2DTranspose(2, kernel_size=4, strides=2, padding="same")(flow)
        up_feat = layers.Conv2DTranspose(2, kernel_size=4, strides=2, padding="same")(c)
        return flow, up_flow, up_feat
    else:
        return flow, c





