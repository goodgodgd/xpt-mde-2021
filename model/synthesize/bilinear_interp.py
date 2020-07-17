import tensorflow as tf
from utils.decorators import shape_check


class BilinearInterpolation:
    @shape_check
    def __call__(self, image, pixel_coords, valid_mask=None):
        """
        :param image: source image [batch, num_src, height, width, ?]
        :param pixel_coords: floating-point pixel coordinates (u,v,1)
                [batch, num_src, 3, height*width] or [batch, num_src, height, width, 2] (is_flow=True)
        :param valid_mask: zero pixels in valid_mask are INVALID in recon_image [batch, height, width, 1]
        :return: reconstructed image [batch, num_src, height, width, ?]
        """
        batch, num_src, height, width, channels = image.get_shape()

        # pixel_floorceil[batch, num_src, :, height*width] = (u_ceil, u_floor, v_ceil, v_floor)
        pixel_floorceil = self.neighbor_int_pixels(pixel_coords, height, width)

        # valid_mask: [batch, num_src, 1, height*width]
        valid_mask2 = self.make_valid_mask(pixel_floorceil, valid_mask, batch)

        # weights[batch, num_src, :, height*width] = (w_uf_vf, w_uf_vc, w_uc_vf, w_uc_vc)
        weights = self.calc_neighbor_weights([pixel_coords, pixel_floorceil, valid_mask2])

        # sampled_image[batch, num_src, :, height, width, 3] =
        # (im_uf_vf, im_uf_vc, im_uc_vf, im_uc_vc)
        sampled_images = self.sample_neighbor_images([image, pixel_floorceil])

        # recon_image[batch, num_src, height*width, 3]
        flat_image = self.merge_images([sampled_images, weights])
        recon_image = tf.reshape(flat_image, shape=(batch, num_src, height, width, channels))
        return recon_image

    def neighbor_int_pixels(self, pixel_coords, height, width):
        """
        :param pixel_coords: (u, v) [batch, num_src, 2, height*width]
        :param height: image height
        :param width: image width
        :return: (u_floor, u_ceil, v_floor, v_ceil) [batch, num_src, 4, height*width]
        """
        u = tf.slice(pixel_coords, (0, 0, 0, 0), (-1, -1, 1, -1))
        u_floor = tf.floor(u)
        u_ceil = tf.clip_by_value(u_floor + 1, 0, width - 1)
        u_floor = tf.clip_by_value(u_floor, 0, width - 1)
        v = tf.slice(pixel_coords, (0, 0, 1, 0), (-1, -1, 1, -1))
        v_floor = tf.floor(v)
        v_ceil = tf.clip_by_value(v_floor + 1, 0, height - 1)
        v_floor = tf.clip_by_value(v_floor, 0, height - 1)
        pixel_floorceil = tf.concat([u_floor, u_ceil, v_floor, v_ceil], axis=2)
        return pixel_floorceil

    @shape_check
    def make_valid_mask(self, pixel_floorceil, valid_mask, batch):
        """
        :param pixel_floorceil: (u_floor, u_ceil, v_floor, v_ceil) (int) [batch, num_src, 4, height*width]
        :param valid_mask: zero pixels in valid_mask are INVALID in recon_image [batch, height, width, 1]
        :param batch: batch size
        :return: mask [batch, num_src, 1, height*width]
        """
        uf = pixel_floorceil[:, :, 0:1, :]
        uc = pixel_floorceil[:, :, 1:2, :]
        vf = pixel_floorceil[:, :, 2:3, :]
        vc = pixel_floorceil[:, :, 3:4, :]
        u_mask = tf.equal(uf + 1, uc)
        v_mask = tf.equal(vf + 1, vc)
        # mask [batch, num_src, 1, height*width]
        mask = tf.logical_and(u_mask, v_mask)

        if valid_mask is not None:
            # nonzero_mask: [batch, height, width, 1] -> [batch, 1, 1, height*width]
            nonzero_mask = tf.reshape(valid_mask, shape=(batch, 1, 1, -1))
            nonzero_mask = tf.math.not_equal(nonzero_mask, 0)
            mask = tf.logical_and(mask, nonzero_mask)

        mask = tf.cast(mask, tf.float32)
        return mask

    @shape_check
    def calc_neighbor_weights(self, inputs):
        pixel_coords, pixel_floorceil, valid_mask = inputs
        """
        pixel_coords: (u, v) (float) [batch, num_src, 2, height*width]
        pixel_floorceil: (u_floor, u_ceil, v_floor, v_ceil) (int) [batch, num_src, 4, height*width]
        valid_mask: [batch, num_src, 1, height*width]
        return: weights of four neighbor pixels (w_uf_vf, w_uf_vc, w_uc_vf, w_uc_vc) 
                [batch, num_src, 4, height*width]
        """
        ui, vi = (0, 1)
        uf, uc, vf, vc = (0, 1, 2, 3)
        w_uf = pixel_floorceil[:, :, uc:uc + 1, :] - pixel_coords[:, :, ui:ui + 1, :]
        w_uc = pixel_coords[:, :, ui:ui + 1, :] - pixel_floorceil[:, :, uf:uf + 1, :]
        w_vf = pixel_floorceil[:, :, vc:vc + 1, :] - pixel_coords[:, :, vi:vi + 1, :]
        w_vc = pixel_coords[:, :, vi:vi + 1, :] - pixel_floorceil[:, :, vf:vf + 1, :]
        w_ufvf = w_uf * w_vf
        w_ufvc = w_uf * w_vc
        w_ucvf = w_uc * w_vf
        w_ucvc = w_uc * w_vc
        weights = tf.concat([w_ufvf, w_ufvc, w_ucvf, w_ucvc], axis=2)
        weights = weights * valid_mask
        return weights

    @shape_check
    def sample_neighbor_images(self, inputs):
        source_image, pixel_floorceil = inputs
        """
        source_image: [batch, num_src, height, width, 3]
        pixel_floorceil: (u_floor, u_ceil, v_floor, v_ceil) [batch, num_src, 4, height*width]
        return: flattened sampled image [(batch, num_src, 4, height*width, 3)]
        """
        pixel_floorceil = tf.cast(pixel_floorceil, tf.int32)
        uf = pixel_floorceil[:, :, 0, :]
        uc = pixel_floorceil[:, :, 1, :]
        vf = pixel_floorceil[:, :, 2, :]
        vc = pixel_floorceil[:, :, 3, :]

        """
        CAUTION: `tf.gather_nd` looks preferable over `tf.gather`
        however, `tf.gather_nd` raises error that some shape is unknown
        `tf.gather_nd` has no problem with eagerTensor (which has specific values)
        but raises error with Tensor (placeholder from tf.keras.layers.Input())
        It seems to be a bug.
        Suprisingly, `tf.gather` works nicely with 'Tensor'
        """
        # tf.stack([vf, uf]): [batch, num_src, height*width, 2(v,u)]
        imflat_ufvf = tf.gather_nd(source_image, tf.stack([vf, uf], axis=-1), batch_dims=2)
        imflat_ufvc = tf.gather_nd(source_image, tf.stack([vc, uf], axis=-1), batch_dims=2)
        imflat_ucvf = tf.gather_nd(source_image, tf.stack([vf, uc], axis=-1), batch_dims=2)
        imflat_ucvc = tf.gather_nd(source_image, tf.stack([vc, uc], axis=-1), batch_dims=2)

        # sampled_images: (batch, num_src, 4, height*width, 3)
        sampled_images = tf.stack([imflat_ufvf, imflat_ufvc, imflat_ucvf, imflat_ucvc], axis=2,
                                  name="stack_samples")
        return sampled_images

    def merge_images(self, inputs):
        sampled_images, weights = inputs
        """
        sampled_images: flattened sampled image [batch, num_src, 4, height*width, 3]
        weights: 4 neighbor pixel weights (w_uf_vf, w_uf_vc, w_uc_vf, w_uc_vc) 
                 [batch, num_src, 4, height*width]
        return: merged_flat_image, [batch, num_src, height*width, 3]
        """
        # expand dimension to channel
        weights = tf.expand_dims(weights, -1)
        weighted_image = sampled_images * weights
        merged_flat_image = tf.reduce_sum(weighted_image, axis=2)
        return merged_flat_image

    @shape_check
    def erase_invalid_pixels(self, inputs):
        flat_image, depth, batch = inputs
        """
        flat_image: [batch, num_src, height*width, 3]
        depth: target view depth [batch, height, width, 1]
        return: [batch, num_src, height*width, 3]
        """
        # depth_vec [batch, height*width, 1]
        depth_vec = tf.reshape(depth, shape=(batch, -1, 1))
        depth_vec = tf.expand_dims(depth_vec, 1)
        # depth_vec [batch, 1, height*width, 1]
        depth_invalid_mask = tf.math.equal(depth_vec, 0)
        flat_image = tf.where(depth_invalid_mask, tf.constant(0, dtype=tf.float32), flat_image)
        return flat_image


class FlowBilinearInterpolation:
    @shape_check
    def __call__(self, image, flow):
        """
        :param image: source image [batch*num_src, height, width, ?]
        :param flow: optical flow [batch*num_src, height, width, 2(u,v)]
        :return: reconstructed image [batch*num_src, height, width, ?]
        """
        _, height, width, channels = image.get_shape()
        # -> [batch*num_src, 1, height, width, ?]
        feature = tf.expand_dims(image, axis=1)
        flow = tf.expand_dims(flow, axis=1)
        print("bilinear feature shape:", feature.get_shape())
        flow = self.flow_to_pixel_coordinates(flow)
        warped_feature = BilinearInterpolation()(feature, flow)
        warped_feature = tf.squeeze(warped_feature, axis=1)
        return warped_feature

    def flow_to_pixel_coordinates(self, flow):
        """
        :param flow: optical flow [batch, num_src, height, width, 2(u,v)]
        :return: pixel_coords: source image pixel coordinates [batch, num_src, 2, height*width]
        """
        batch, num_src, height, width, _ = flow.get_shape()
        u = tf.range(0, width, 1, dtype=tf.float32)
        v = tf.range(0, height, 1, dtype=tf.float32)
        ugrid, vgrid = tf.meshgrid(u, v)
        uvgrid = tf.stack([ugrid, vgrid], axis=0)
        # uvgrid -> [1, 1, 2, height*width]
        uvgrid = tf.reshape(uvgrid, (1, 1, 2, -1))

        # uvflow -> [batch, num_src, height*width, 2]
        uvflow = tf.reshape(flow, (batch, num_src, -1, 2))
        # uvflow -> [batch, num_src, 2, height*width]
        uvflow = tf.transpose(uvflow, perm=[0, 1, 3, 2])

        # add flow to basic grid
        pixel_coords = uvgrid - uvflow
        return pixel_coords

