import tensorflow as tf
from config import opts
from utils.util_class import WrongInputException

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


def augmentation_factory(augment_probs=opts.AUGMENT_PROBS):
    augmenters = []
    for key, prob in augment_probs.items():
        if key is "CropAndResize":
            augm = CropAndResize(prob)
        elif key is "HorizontalFlip":
            augm = HorizontalFlip(prob)
        elif key is "ColorJitter":
            augm = ColorJitter(prob)
        else:
            raise WrongInputException(f"Wrong augmentation type: {key}")
        augmenters.append(augm)
    total_augment = TotalAugment(augmenters)
    return total_augment


class TotalAugment:
    def __init__(self, augment_objects):
        self.augment_objects = augment_objects

    def __call__(self, features):
        features = self.augment_oneside(features)
        if "image_R" in features:
            print("right side sugment")
            features = self.augment_oneside(features, "_R")
        return features

    def augment_oneside(self, features, suffix=""):
        features = Preprocess()(features, suffix)
        for augmenter in self.augment_objects:
            features = augmenter(features, suffix)
        features = Postprocess()(features, suffix)
        return features


class AugmentBase:
    def __init__(self, aug_prob=0.):
        self.aug_prob = aug_prob
        self.param = 0

    def __call__(self, features, suffix):
        raise NotImplementedError()


class Preprocess(AugmentBase):
    def __init__(self):
        super().__init__()

    def __call__(self, features, suffix=""):
        """
        :return: append features below
            image_aug: [batch*snippet, height, width, 3]
            depth_gt_aug: [batch, height, width, 1]
            intrinsic_aug: [batch, 3, 3]
            pose_gt_aug: [batch, numsrc, 4, 4]
        """
        image5d = features["image5d" + suffix]
        batch, snippet, height, width, channels = image5d.get_shape()
        numsrc = snippet - 1

        # to use tf.image functions, reshape to [batch*snippet, height, width, 3]
        image_aug = tf.reshape(image5d, (batch*snippet, height, width, channels))
        features["image_aug" + suffix] = image_aug
        # copy intrinsic
        features["intrinsic_aug" + suffix] = tf.reshape(features["intrinsic" + suffix], (batch, 3, 3))
        # copy pose_gt
        if "pose_gt" + suffix in features:
            features["pose_gt_aug" + suffix] = tf.reshape(features["pose_gt" + suffix], (batch, numsrc, 4, 4))
        # copy depth_gt
        if "depth_gt" + suffix in features:
            features["depth_gt_aug" + suffix] = tf.reshape(features["depth_gt" + suffix], (batch, height, width, 1))
        return features


class Postprocess(AugmentBase):
    def __init__(self):
        super().__init__()

    def __call__(self, features, suffix=""):
        """
        :return: append features below
            image_aug: [batch, snippet, height, width, 3] <- only change!
            depth_gt_aug: [batch, height, width, 1]
            intrinsic_aug: [batch, 3, 3]
            pose_gt_aug: [batch, numsrc, 4, 4]
        """
        image5d = features["image5d" + suffix]
        batch, snippet, height, width, channels = image5d.get_shape()
        imkey = "image_aug" + suffix
        features[imkey] = tf.reshape(features[imkey], (batch, snippet, height, width, channels))
        return features


class CropAndResize(AugmentBase):
    """
    randomly crop "image_aug" and resize it to original size
    create "intrinsic_aug" as camera matrix for "image_aug"
    """
    def __init__(self, aug_prob=0.3):
        super().__init__(aug_prob)
        self.half_crop_ratio = 0.1

    def __call__(self, features, suffix=""):
        batch, snippet, height, width, channels = features["image5d"].get_shape()
        image = features["image_aug" + suffix]
        crop_size = tf.constant([height, width])
        num_box = batch*snippet
        box_indices = tf.range(0, num_box)
        boxes = self.random_crop_boxes(num_box)
        self.param = boxes[0]

        image_aug = tf.image.crop_and_resize(image, boxes, box_indices, crop_size)
        features["image_aug" + suffix] = image_aug
        intrin_aug = self.adjust_intrinsic(features["intrinsic_aug" + suffix], boxes, crop_size)
        features["intrinsic_aug" + suffix] = intrin_aug

        depth_key = "depth_gt_aug" + suffix
        if depth_key in features:
            batch_size = features[depth_key].get_shape()[0]
            depth_aug = tf.image.crop_and_resize(features[depth_key], boxes[:batch_size], box_indices[:batch_size],
                                                 crop_size, method="nearest")
            features[depth_key] = depth_aug

        return features

    def random_crop_boxes(self, num_box):
        # aug_prob : 1-aug_prob = half_crop_ratio : minval1
        maxval1 = self.half_crop_ratio
        minval1 = -(1. - self.aug_prob) * self.half_crop_ratio / self.aug_prob
        y1x1 = tf.random.uniform((1, 2), minval1, maxval1)
        y1x1 = tf.clip_by_value(y1x1, 0, 1)
        minval2 = 1. - maxval1
        maxval2 = 1. - minval1
        y2x2 = tf.random.uniform((1, 2), minval2, maxval2)
        y2x2 = tf.clip_by_value(y2x2, 0, 1)
        assert (minval1 < maxval1) and (minval2 < maxval2)
        # boxes: [1, 4]
        boxes = tf.concat([y1x1, y2x2], axis=1)
        # boxes: [num_box, 4]
        boxes = tf.tile(boxes, [num_box, 1])
        return boxes

    def adjust_intrinsic(self, intrinsic, boxes, imsize):
        """
        :param intrinsic: [batch, 3, 3]
        :param boxes: (y1,x1,y2,x2) in range [0~1] [batch, 4]
        :param imsize: [height, width] [2]
        :return: adjusted intrinsic [batch, 3, 3]
        """
        imsize = tf.cast(imsize, tf.float32)
        # size: [1, 3, 3], contents: [[0, 0, x1_ratio*width], [0, 0, y1_ratio*height], [0, 0, 0]]
        center_change = tf.stack([tf.concat([0., 0., boxes[0, 1]*imsize[1]], axis=0),
                                  tf.concat([0., 0., boxes[0, 0]*imsize[0]], axis=0),
                                  tf.concat([0., 0., 0.], axis=0)],
                                 axis=0)
        # cx'=cx-x1, cy'=cy-y1
        intrin_crop = intrinsic - center_change
        # cx,fx *= W/(x2-x1),  cy,fy *= H/(y2-y1)
        x_ratio = 1. / (boxes[0, 3] - boxes[0, 1])
        y_ratio = 1. / (boxes[0, 2] - boxes[0, 0])
        intrin_adj = tf.stack([intrin_crop[:, 0] * x_ratio, intrin_crop[:, 1] * y_ratio, intrin_crop[:, 2]], axis=1)
        return intrin_adj


class HorizontalFlip(AugmentBase):
    """
    randomly horizontally flip "image_aug" by aug_prob
    """
    def __init__(self, aug_prob=0.2):
        super().__init__(aug_prob)

    def __call__(self, features, suffix=""):
        image = features["image_aug" + suffix]
        intrinsic = features["intrinsic_aug" + suffix]
        pose = features["pose_gt_aug" + suffix]
        rndval = tf.random.uniform(())

        image_flip, intrin_flip, pose_flip = \
            tf.cond(rndval < self.aug_prob,
                    lambda: self.flip_features(image, intrinsic, pose),
                    lambda: (image, intrinsic, pose)
                    )

        features["image_aug" + suffix] = image_flip
        features["intrinsic_aug" + suffix] = intrin_flip
        features["pose_gt_aug" + suffix] = pose_flip

        depth_key = "depth_gt_aug" + suffix
        if depth_key in features:
            depth_aug = tf.cond(rndval < self.aug_prob,
                                lambda: tf.image.flip_left_right(features[depth_key]),
                                lambda: features[depth_key])
            features[depth_key] = depth_aug

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


class ColorJitter(AugmentBase):
    def __init__(self, aug_prob=0.2):
        super().__init__(aug_prob)

    def __call__(self, features, suffix=""):
        image = features["image_aug" + suffix]
        rndval = tf.random.uniform(())
        # srcimg = image[0, 0:100:20, 0:300:60, 1].numpy()

        image_jit, param = tf.cond(rndval < self.aug_prob,
                                   lambda: self.jitter_color(image),
                                   lambda: (image, tf.constant([0, 0], dtype=tf.float32)))
        self.param = param
        features["image_aug" + suffix] = image_jit
        # dstimg = image_jit[0, 0:100:20, 0:300:60, 1].numpy()
        # print("image jit diff:", dstimg - srcimg)
        return features

    def jitter_color(self, image):
        image = (image + 1.) / 2.
        gamma = tf.random.uniform((), minval=0.5, maxval=1.5)
        saturation = tf.random.uniform((), minval=0.5, maxval=1.5)
        param = tf.concat([gamma, saturation], axis=0)
        image = tf.image.adjust_saturation(image, saturation)
        image = tf.image.adjust_gamma(image, gamma=gamma, gain=1.)
        image = image * 2. - 1.
        return image, param


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
    boxes = tf.tile(tf.constant([[ycrop, xcrop, 1-ycrop, 1-xcrop]]), [batch, 1])
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
    print("pose vec:\n", pose_vec[1])
    print("pose mat:\n", pose_mat[1, 1])
    print("pose mat flip:\n", pose_mat_flip[1, 1])
    print("pose vec flip:\n", pose_vec_flip[1])
    print("pose vec rotation [batch, numsrc]: (rad)\n", np.linalg.norm(pose_vec[:, :, 3:], axis=1))
    print("pose vec flip rotation [batch, numsrc]: (rad)\n", np.linalg.norm(pose_vec_flip[:, :, 3:], axis=1))
    print("pose == pose_flip:\n", np.isclose(pose_vec[1, 1], pose_vec_flip[1, 1]))

    flip_vec = tf.constant([[[[-1, 1, 1, 1, -1, -1]]]], dtype=tf.float32)
    assert np.isclose(pose_vec.numpy(), pose_vec_flip.numpy()*flip_vec, atol=1.e-3).all(), \
        f"{pose_vec.numpy() - pose_vec_flip.numpy()*flip_vec}"
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


import os.path as op
import cv2
from config import opts
from tfrecords.tfrecord_reader import TfrecordGenerator
from utils.util_funcs import to_uint8_image, multi_scale_depths
from model.synthesize.synthesize_base import SynthesizeMultiScale
from utils.convert_pose import pose_matr2rvec_batch


def test_augmentations():
    print("===== test test_augmentations")
    tfrgen = TfrecordGenerator(op.join(opts.DATAPATH_TFR, "kitti_raw_test"), shuffle=False)
    dataset = tfrgen.get_generator()
    data_aug = {"Preprocess": Preprocess(),
                "CropAndResize": CropAndResize(aug_prob=0.5),
                "HorizontalFlip": HorizontalFlip(aug_prob=0.5),
                "ColorJitter": ColorJitter(aug_prob=0.5),
                "Postprocess": Postprocess()}

    for bi, features in enumerate(dataset):
        print(f"\n!!~~~~~~~~~~ {bi}: new features ~~~~~~~~~~!!")
        images = []
        for name, augment in data_aug.items():
            features = augment(features)
            img = show_result(features, name, augment.param)
            images.append(img)

        source_image, synth_target = synthesize_target(features, "_aug")
        images.append(synth_target)
        images.append(source_image)
        images = np.concatenate(images, axis=0)
        cv2.imshow("augmentation", images)

        ori_images = []
        raw_image_u8 = to_uint8_image(features["image"])
        ori_images.append(raw_image_u8[0, -opts.IM_HEIGHT:])
        source_image, synth_target = synthesize_target(features)
        ori_images.append(synth_target)
        ori_images.append(source_image)
        ori_images = np.concatenate(ori_images, axis=0)
        cv2.imshow("original image", ori_images)

        key = cv2.waitKey()
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


def show_result(features, name, param):
    batch, snippet, height, width, chann = features["image5d"].get_shape()
    numsrc = snippet - 1

    print(f"----- augmentation: {name}")
    print("parameter:", param)
    print("image shape", features["image_aug"].get_shape())
    image = tf.reshape(features["image_aug"], (batch*snippet, height, width, chann))
    image_u8 = to_uint8_image(image)
    target = image_u8[numsrc].numpy()
    intrin = features["intrinsic_aug"]
    print("intrinsic:\n", intrin[0].numpy())
    pose = features["pose_gt_aug"]
    print("pose:\n", pose[0, 0].numpy())
    return target


def synthesize_target(features, suffix=""):
    sources, target, intrinsic, depth_gt_ms, pose_gt = prep_synthesize(features, suffix)
    synth_target_ms = SynthesizeMultiScale()(sources, intrinsic, depth_gt_ms, pose_gt)
    synth_u8 = to_uint8_image(synth_target_ms[0])
    synth_u8 = synth_u8[0, 0].numpy()
    height = synth_u8.shape[0]
    source_u8 = to_uint8_image(sources)
    source_u8 = source_u8[0, :height].numpy()
    return source_u8, synth_u8


def prep_synthesize(features, suffix):
    batch, snippet, height, width, chann = features["image5d"].get_shape()
    numsrc = snippet - 1
    if suffix is "_aug":
        image = features["image" + suffix]
    else:
        image = features["image5d"]

    image_5d = tf.reshape(image, (batch, snippet, height, width, chann))
    sources = tf.reshape(image_5d[:, :numsrc], (batch, numsrc*height, width, chann))
    target = image_5d[:, numsrc]
    intrinsic = features["intrinsic" + suffix]
    pose_gt = features["pose_gt" + suffix]
    pose_gt = pose_matr2rvec_batch(pose_gt)
    depth_gt = features["depth_gt" + suffix]
    depth_gt_ms = multi_scale_depths(depth_gt, [1, 2, 4, 8])
    return sources, target, intrinsic, depth_gt_ms, pose_gt


def test_augmentation_factory():
    print("===== test test_augmentations")
    tfrgen = TfrecordGenerator(op.join(opts.DATAPATH_TFR, "kitti_raw_test"), shuffle=False)
    dataset = tfrgen.get_generator()
    augmenter = augmentation_factory(opts.AUGMENT_PROBS)

    for bi, features in enumerate(dataset):
        print(f"\n!!~~~~~~~~~~ {bi}: new features ~~~~~~~~~~!!")
        print("before augment features:")
        fkeys = list(features.keys())
        for i in range(len(features.keys()) // 5):
            print(fkeys[i*5:(i+1)*5])

        features = augmenter(features)

        print("after augment features:")
        fkeys = list(features.keys())
        for i in range(np.ceil(len(features.keys())/5.).astype(int)):
            print(fkeys[i*5:(i+1)*5])

        image = to_uint8_image(features["image5d_R"])
        image_aug = to_uint8_image(features["image_aug_R"])
        batch, snippet, height, width, chann = image.get_shape()
        image = image.numpy()[1].reshape(-1, width, chann)
        image_aug = image_aug.numpy()[1].reshape(-1, width, chann)
        image = np.concatenate([image, image_aug], axis=1)
        cv2.imshow("image vs augmented", image)
        cv2.waitKey(1000)
        key = cv2.waitKey()
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_random_crop_boxes()
    test_adjust_intrinsic()
    test_flip_pose_np()
    test_flip_pose_tf()
    test_flip_intrinsic()
    test_augmentations()
    test_augmentation_factory()






