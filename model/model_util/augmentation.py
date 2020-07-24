import tensorflow as tf

"""
depthnet : only target
posenet : full stack [... 3*snippet]
flownet : target [batch] sources [batch*numsrc]
loss : target [batch]
SynthesizeSingleScale: sources [batch, numsrc]
FlowWarpMultiScale: sources [batch, numsrc]

stacked image -> batch*snippet -> augmentation  -> "image" [batch*snippet]
-> "target" [batch] "sources" [batch, numsrc]
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
        """
        :return: features below
            target: [batch, height, width, 3]
            sources: [batch, numsrc, height, width, 3]
            image_aug: [batch, snippet, height, width, 3]
            intrinsic_aug: [batch, 3, 3]
            pose_gt_aug: [batch, numsrc, 4, 4]
        """
        suffix = self.suffix
        image = features["image" + suffix]
        batch, h_stacked, width, channels = image.get_shape()
        height = h_stacked // self.snippet_len
        image_5d = tf.reshape(image, (batch, self.snippet_len, height, width, channels))

        # [batch, height, width, 3]
        features["target" + suffix] = image_5d[:, self.snippet_len-1]
        # [batch, numsrc, height, width, 3]
        features["sources" + suffix] = image_5d[:, :self.snippet_len-1]
        # [batch*snippet, height, width, 3]
        image_aug = tf.reshape(image_5d, (batch*self.snippet_len, height, width, channels))
        features["image_aug" + suffix] = image_aug
        # copy intrinsic
        features["intrinsic_aug" + suffix] = tf.reshape(features["intrinsic" + suffix], (batch, 3, 3))
        # copy pose_gt
        features["pose_gt_aug" + suffix] = tf.reshape(features["pose_gt" + suffix], (batch, 4, 4))
        return features


class CropAndResize:
    """
    randomly crop "image_aug" and resize it to original size
    create "intrinsic_aug" as camera matrix for "image_aug"
    """
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

        intrin_aug = self.adjust_intrinsic(features["intrinsic_aug" + suffix], boxes, crop_size)
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
    """
    randomly horizontally flip "image_aug" by flip_prob
    """
    def __init__(self, suffix=""):
        self.flip_prob = 0.2
        self.suffix = suffix

    def __call__(self, features):
        suffix = self.suffix
        # (batch*snippet, height, width, 3)
        image = features["image_aug" + suffix]
        intrinsic = features["intrinsic_aug" + suffix]
        pose = features["pose_gt_aug" + suffix]
        rndval = tf.random.uniform(())

        image_flip, intrin_flip, pose_flip = \
            tf.cond(rndval < self.flip_prob,
                    self.flip_features(image, intrinsic, pose),
                    lambda: (image_flip, intrin_flip, pose_flip)
                    )

        features["image_aug" + suffix] = image_flip
        features["intrinsic_aug" + suffix] = intrin_flip
        features["pose_gt_aug" + suffix] = pose_flip
        return features

    def flip_features(self, image, intrinsic, pose):
        """
        :param image: [batch, height, width,
        :param intrinsic:
        :param pose:
        :return:
        """
        image_flip = tf.image.flip_left_right(image)
        intrin_flip = self.flip_intrinsic(intrinsic, image.get_shape())
        pose_flip = self.flip_pose(pose)
        return image_flip, intrin_flip, pose_flip

    def flip_intrinsic(self, intrinsic, imshape):
        batch, height, width, _ = imshape
        intrin_wh = tf.constant([[[0, 0, width], [0, 0, 0], [0, 0, 0]]], dtype=tf.float32)
        intrin_flip = tf.abs(intrin_wh - intrinsic)
        return intrin_flip

    def flip_pose(self, pose):
        T_flip = tf.constant([[[[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]]], dtype=tf.float32)
        # [batch, numsrc, 4, 4] = [1, 1, 4, 4] @ [batch, numsrc, 4, 4] @ [1, 1, 4, 4]
        pose_flip = T_flip @ pose @ tf.linalg.inv(T_flip)
        return pose_flip


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
    imsize = tf.constant([height, width], dtype=tf.float32)
    intrinsic = tf.constant([[[width/2, 0, width/2], [0, height/2, height/2], [0, 0, 1]]], dtype=tf.float32)
    intrinsic = tf.tile(intrinsic, [batch, 1, 1])
    print("intrinsic original", intrinsic[0])

    xcrop, ycrop = 0.05, 0.1
    cropper = CropAndResize()
    boxes = tf.tile(tf.constant([[xcrop, ycrop, 1-xcrop, 1-ycrop]]), [batch, 1])
    print("crop box:", boxes[0])

    # EXECUTE
    intrin_adj = cropper.adjust_intrinsic(intrinsic, boxes, imsize)
    print("intrinsic adjusted", intrin_adj[0])

    assert np.isclose(intrin_adj.numpy()[0], intrin_adj.numpy()[-1]).all()
    assert np.isclose(intrin_adj[0, 0, 0], width / 2 / (1 - 2*xcrop)), \
           f"fx={intrin_adj[0, 0, 0]}, expected={width / 2 / (1 - 2*xcrop)}"
    assert np.isclose(intrin_adj[0, 0, 2], width / 2), \
           f"cx={intrin_adj[0, 0, 2]}, expected={width / 2}"
    print("!!! test_adjust_intrinsic passed")


def test_flip_pose_np():
    print("===== test test_flip_pose_np")
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
    print("pose vec rotation: (rad)\n", np.linalg.norm(pose_vec[:, 3:], axis=1))
    print("pose vec flip rotation: (rad)\n", np.linalg.norm(pose_vec_flip[:, 3:], axis=1))
    print("pose == pose_flip:\n", np.isclose(pose_vec, pose_vec_flip))

    flip_vec = np.array([[-1, 1, 1, 1, -1, -1]], dtype=np.float32)
    assert np.isclose(pose_vec, pose_vec_flip*flip_vec).all()
    print("!!! test_flip_pose_np passed")


def test_flip_pose_tf():
    print("===== test test_flip_pose_tf")
    batch, numsrc = 2, 2
    pose_vec = tf.random.uniform((batch, numsrc, 6), -2, 2)
    pose_mat = cp.pose_rvec2matr_batch(pose_vec)
    flipper = HorizontalFlip()
    pose_mat_flip = flipper.flip_pose(pose_mat)
    pose_vec_flip = cp.pose_matr2rvec_batch(pose_mat_flip)
    print("pose vec:\n", pose_vec[1, 1])
    print("pose mat:\n", pose_mat[1, 1])
    print("pose mat flip:\n", pose_mat_flip[1, 1])
    print("pose vec flip:\n", pose_vec_flip[1, 1])
    print("pose vec rotation [batch, numsrc]: (rad)\n", np.linalg.norm(pose_vec[:, :, 3:], axis=1))
    print("pose vec flip rotation [batch, numsrc]: (rad)\n", np.linalg.norm(pose_vec_flip[:, :, 3:], axis=1))
    print("pose == pose_flip:\n", np.isclose(pose_vec[1, 1], pose_vec_flip[1, 1]))

    flip_vec = tf.constant([[[[-1, 1, 1, 1, -1, -1]]]], dtype=tf.float32)
    assert np.isclose(pose_vec.numpy(), pose_vec_flip.numpy()*flip_vec).all()
    print("!!! test_flip_pose_tf passed")


def test_flip_intrinsic():
    print("===== test test_flip_intrinsic")
    batch, height, width = 3, 200, 240
    intrinsic = tf.random.uniform((batch, 3, 3), minval=100, maxval=200)
    print("intrinsic original", intrinsic[0])
    imshape = (batch, height, width, 3)
    flipper = HorizontalFlip()

    # EXECUTE
    intrin_flip = flipper.flip_intrinsic(intrinsic, imshape)

    intrinsic = intrinsic.numpy()
    intrin_flip = intrin_flip.numpy()
    # fy, cy: SAME
    assert np.isclose(intrinsic[:, 1:], intrin_flip[:, 1:]).all(), \
        f"original\n{intrinsic[:, 1:]}\nflipped\n{intrin_flip[:, 1:]}"
    # fx: SAME
    assert np.isclose(intrinsic[:, 0, :2], intrin_flip[:, 0, :2]).all(), \
        f"original\n{intrinsic[:, 0, :2]}\nflipped\n{intrin_flip[:, 0, :2]}"
    # cx <- W - cx
    assert np.isclose(width - intrinsic[:, 0, 2], intrin_flip[:, 0, 2]).all(), \
        f"original\n{intrinsic[:, 0, 2]}\nflipped\n{intrin_flip[:, 0, 2]}"
    print("horizontally flipped intrinsic\n", intrin_flip[0])
    print("!!! test_flip_intrinsic passed")


if __name__ == "__main__":
    test_random_crop_boxes()
    test_adjust_intrinsic()
    test_flip_pose_np()
    test_flip_pose_tf()
    test_flip_intrinsic()






