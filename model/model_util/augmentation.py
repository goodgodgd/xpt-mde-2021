import tensorflow as tf

"""
depthnet : only target
posenet : full stack [... 3*snippet]
flownet : target [batch] sources [batch*num_src]
loss : target [batch]
SynthesizeSingleScale: sources [batch, num_src]
FlowWarpMultiScale: sources [batch, num_src]

stacked image -> batch*snippet -> augmentation  -> "image" [batch*snippet]
-> "target" [batch] "sources" [batch, num_src]
"""


class TotalAugment:
    def __init__(self, augment_objects):
        self.augment_objects = augment_objects

    def __call__(self, features):
        for augmenter in self.augment_objects:
            features = augmenter(features)


class Preprocess:
    def __init__(self, snippet_len, suffix=""):
        self.snippet_len = snippet_len
        self.suffix = suffix

    def __call__(self, features):
        suffix = self.suffix
        image = features["image" + suffix]
        batch, h_stacked, width, channels = image.get_shape()
        height = h_stacked // self.snippet_len
        image_5d = tf.reshape(image, (batch, self.snippet_len, height, width, channels))

        # [batch, height, width, 3]
        features["target" + suffix] = image_5d[:, self.snippet_len-1]
        # [batch, num_src, height, width, 3]
        features["sources" + suffix] = image_5d[:, :self.snippet_len-1]
        # [batch*snippet, height, width, 3]
        image_bxs = tf.reshape(image_5d, (batch*self.snippet_len, height, width, channels))
        features["image_aug"] = image_bxs
        return features


class CropAndResize:
    def __init__(self, suffix=""):
        self.half_crop_ratio = 0.1
        self.crop_prob = 0.3
        self.suffix = suffix

    def __call__(self, features):
        suffix = self.suffix
        # [batch*snippet, height, width, 3]
        image = features["image_aug" + suffix]
        batch, height, width, _ = image.get_shape()

        crop_size = tf.constant([height, width])
        box_indices = tf.range(0, batch)
        boxes = self.random_crop_boxes(batch)
        image_aug = tf.image.crop_and_resize(image, boxes, box_indices, crop_size)
        features["image_aug" + suffix] = image_aug

        intrin_aug = self.adjust_intrinsic(features["intrinsic" + suffix], boxes, crop_size)
        features["intrinsic_aug" + suffix] = intrin_aug
        return features

    def random_crop_boxes(self, num_box):
        # crop_prob : 1-crop_prob = half_crop_ratio : x
        maxval1 = self.half_crop_ratio
        minval1 = -(1. - self.crop_prob) * self.half_crop_ratio / self.crop_prob
        x1y1 = tf.random.uniform((1, 2), minval1, maxval1)
        x1y1 = tf.clip_by_value(x1y1, 0, 1)
        minval2 = 1. - maxval1
        maxval2 = 1. - minval1
        x2y2 = tf.random.uniform((1, 2), minval2, maxval2)
        x2y2 = tf.clip_by_value(x2y2, 0, 1)
        assert (minval1 < maxval1) and (minval2 < maxval2)
        # boxes: [1, 4]
        boxes = tf.concat([x1y1, x2y2], axis=1)
        # boxes: [num_box, 4]
        boxes = tf.tile(boxes, [num_box, 1])
        return boxes

    def adjust_intrinsic(self, intrinsic, boxes, imsize):
        """
        :param intrinsic: [batch, 3, 3]
        :param boxes: (x1,y1,x2,y2) in range [0~1] [batch, 4]
        :param imsize: [height, width] [2]
        :return: adjusted intrinsic [batch, 3, 3]
        """
        # size: [1, 3, 3], contents: [[0, 0, x1], [0, 0, y1], [0, 0, 0]]
        center_change = tf.stack([tf.concat([0, 0, boxes[0, 0]*imsize[1]], axis=0),
                                  tf.concat([0, 0, boxes[0, 1]*imsize[0]], axis=0),
                                  tf.concat([0., 0., 0.], axis=0)],
                                 axis=0)
        # cx'=cx-x1, cy'=cy-y1
        intrin_crop = intrinsic - center_change
        # cx,fx *= W/(x2-x1),  cy,fy *= H/(y2-y1)
        x_ratio = 1. / (boxes[0, 2] - boxes[0, 0])
        y_ratio = 1. / (boxes[0, 3] - boxes[0, 1])
        intrin_adj = tf.stack([intrin_crop[:, 0] * x_ratio, intrin_crop[:, 1] * y_ratio, intrin_crop[:, 2]], axis=1)
        return intrin_adj


class HorizontalFlip:
    def __init__(self):
        self.flip_prob = 0.2

    def __call__(self, features):
        # (batch*snippet, height, width, 3)
        image = features["image_aug"]
        batch, height, width, _ = image.get_shape()
        flipped = tf.image.flip_left_right(image)


class ColorJitter:
    def __call__(self, features):
        # saturated = tf.image.adjust_saturation(image, 3)
        # bright = tf.image.adjust_brightness(image, 0.4)
        pass


# ---------------------------------
import numpy as np
import utils.convert_pose as cp


def test_random_crop_boxes():
    print("===== test test_random_crop_boxes")
    cropper = CropAndResize()
    boxes = cropper.random_crop_boxes(4)
    print("boxes:", boxes)
    wh = boxes[:, 2:] - boxes[:, :2]
    assert (wh.numpy() > cropper.half_crop_ratio*2).all()
    print("!!! test_random_crop_boxes passed")


def test_adjust_intrinsic():
    print("===== test test_adjust_intrinsic")
    batch, height, width = 3, 200, 240
    xcrop, ycrop = 0.05, 0.1
    cropper = CropAndResize()
    boxes = tf.tile(tf.constant([[xcrop, ycrop, 1-xcrop, 1-ycrop]]), [batch, 1])
    print("crop box:", boxes[0])
    intrinsic = tf.constant([[[width/2, 0, width/2], [0, height/2, height/2], [0, 0, 1]]], dtype=tf.float32)
    intrinsic = tf.tile(intrinsic, [batch, 1, 1])
    print("intrinsic original", intrinsic[0])
    imsize = tf.constant([height, width], dtype=tf.float32)
    intrin_adj = cropper.adjust_intrinsic(intrinsic, boxes, imsize)
    print("intrinsic adjusted", intrin_adj[0])

    assert np.isclose(intrin_adj.numpy()[0], intrin_adj.numpy()[-1]).all()
    assert np.isclose(intrin_adj[0, 0, 0], width / 2 / (1 - 2*xcrop)), \
           f"fx={intrin_adj[0, 0, 0]}, expected={width / 2 / (1 - 2*xcrop)}"
    assert np.isclose(intrin_adj[0, 0, 2], width / 2), \
           f"cx={intrin_adj[0, 0, 2]}, expected={width / 2}"
    print("!!! test_adjust_intrinsic passed")


def test_flip_pose():
    print("===== test test_flip_pose")
    batch = 2
    pose_vec = np.random.uniform(-2, 2, (batch, 6))
    pose_mat = cp.pose_rvec2matr(pose_vec)
    flip = np.identity(4)
    flip[0, 0] = -1
    flip = flip[np.newaxis, ...]
    pose_mat_flip = np.matmul(np.matmul(flip, pose_mat), np.linalg.inv(flip))
    pose_vec_flip = cp.pose_matr2rvec(pose_mat_flip)
    print("pose vec:\n", pose_vec)
    print("pose mat:\n", pose_mat)
    print("pose mat flip:\n", pose_mat_flip)
    print("pose vec flip:\n", pose_vec_flip)
    print("pose vec rotation:\n", np.linalg.norm(pose_vec[:, 3:], axis=1))
    print("pose vec flip rotation:\n", np.linalg.norm(pose_vec_flip[:, 3:], axis=1))
    print("pose == pose_flip:\n", np.isclose(pose_vec, pose_vec_flip))

    flip_vec = np.array([[-1, 1, 1, 1, -1, -1]], dtype=np.float32)
    assert np.isclose(pose_vec, pose_vec_flip*flip_vec).all()
    print("!!! test_flip_pose passed")


if __name__ == "__main__":
    test_random_crop_boxes()
    test_adjust_intrinsic()
    test_flip_pose()
















