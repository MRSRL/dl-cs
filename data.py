"""Util for data management."""
import os
import glob
import random
import tensorflow as tf
import numpy as np
import logging
import data_prep
from utils import cfl
from utils import tfmri
from utils import mri
from scipy.stats import ortho_group

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT, None))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def prepare_filenames(dir_name, search_str='/*.tfrecords'):
    """Find and return filenames."""
    if not tf.gfile.Exists(dir_name) or not tf.gfile.IsDirectory(dir_name):
        raise FileNotFoundError('Could not find folder {}'.format(dir_name))

    full_path = os.path.join(dir_name)
    case_list = glob.glob(full_path + search_str)
    random.shuffle(case_list)

    return case_list


def load_masks_cfl(filenames, image_shape=None):
    """Read masks from files."""
    if image_shape is None:
        # First find masks shape...
        image_shape = [0, 0]
        for f in filenames:
            f_cfl = os.path.splitext(f)[0]
            mask = np.squeeze(cfl.read(f_cfl))
            shape_z = mask.shape[-2]
            shape_y = mask.shape[-1]
            if image_shape[-2] < shape_z:
                image_shape[-2] = shape_z
            if image_shape[-1] < shape_y:
                image_shape[-1] = shape_y

    masks = np.zeros([len(filenames)] + image_shape, dtype=np.complex64)

    i_file = 0
    for f in filenames:
        f_cfl = os.path.splitext(f)[0]
        tmp = np.squeeze(cfl.read(f_cfl))
        tmp = mri.zeropad(tmp, image_shape)
        masks[i_file, :, :] = tmp
        i_file = i_file + 1

    return masks


def prep_tfrecord(example, masks,
                  out_shape=[80, 180],
                  init_shape=None,
                  bandpass_pad=0,
                  shape_calib=20,
                  shape_scale=5,
                  scale_factor=1,
                  num_channels=6, num_maps=2,
                  shuffle_channels=True,
                  resize_sensemaps=False,
                  random_seed=0):
    """Prepare tfrecord for training"""
    name = 'prep_tfrecord'
    logger.info('Preparing tfrecords...')

    _, xslice, ks_x, sensemap_x, shape_c = data_prep.process_tfrecord(
        example, num_channels=num_channels, num_maps=num_maps)

    if init_shape is None:
        init_shape = out_shape

    ks_x = tf.transpose(ks_x, [1, 2, 0])
    sensemap_x = tf.transpose(sensemap_x, [1, 2, 0])
    if masks is not None:
        # Randomly select mask
        mask_x = tf.constant(masks, dtype=tf.complex64)
        mask_x = tf.random_shuffle(mask_x)
        mask_x = tf.slice(mask_x, [0, 0, 0], [1, -1, -1])
        # Augment sampling masks
        mask_x = tf.image.random_flip_up_down(mask_x, seed=random_seed)
        mask_x = tf.image.random_flip_left_right(mask_x, seed=random_seed)

        # Tranpose to store data as (kz, ky, channels)
        mask_x = tf.transpose(mask_x, [1, 2, 0])
    else:
        mask_x = tf.ones(out_shape + [1], dtype=tf.complex64)

    ks_x = tf.image.flip_up_down(ks_x)
    sensemap_x = tf.image.flip_up_down(sensemap_x)

    # Initially set image size to be all the same
    ks_x = tf.image.resize_image_with_crop_or_pad(
        ks_x,
        init_shape[0] + 2 * bandpass_pad,
        init_shape[1] + 2 * bandpass_pad)
    mask_x = tf.image.resize_image_with_crop_or_pad(
        mask_x,
        init_shape[0] + 2 * bandpass_pad,
        init_shape[1] + 2 * bandpass_pad)

    if shape_calib > 0:
        with tf.name_scope('CalibRegion'):
            logger.info('  Including calib region ({}, {})...'.format(
                shape_calib, shape_calib))
            mask_calib = tf.ones([shape_calib, shape_calib, 1],
                                 dtype=tf.complex64)
            mask_calib = tf.image.resize_image_with_crop_or_pad(
                mask_calib,
                init_shape[0] + 2 * bandpass_pad,
                init_shape[1] + 2 * bandpass_pad)
            mask_x = mask_x * (1 - mask_calib) + mask_calib

    mask_recon = tf.abs(ks_x) / tf.reduce_max(tf.abs(ks_x))
    mask_recon = tf.cast(mask_recon > 1e-7, dtype=tf.complex64)
    mask_x = mask_x * mask_recon

    if shape_scale > 0:
        logger.info('  Scaling ({})...'.format(shape_scale))
        # Assuming calibration region is fully sampled
        scale = tf.image.resize_image_with_crop_or_pad(
            ks_x, shape_scale, shape_scale)
        scale = (tf.reduce_mean(tf.square(tf.abs(scale))) *
                 (shape_scale * shape_scale / out_shape[0] / out_shape[1]))
        scale = tf.cast(1.0 / tf.sqrt(scale), dtype=tf.complex64)
    else:
        logger.info('  Turn off scaling...')
        scale = tf.sqrt(shape_c / num_channels)
        scale = tf.cast(scale, dtype=tf.complex64)

    if scale_factor > 1:
        logger.info('  Extra scale factor {}'.format(scale_factor))
    ks_x = ks_x * scale * scale_factor

    if resize_sensemaps:
        logger.info('  Resizing sensemaps to: ({}, {})'.format(
            out_shape[0], out_shape[1]))
        sensemap_x = tfmri.complex_to_channels(sensemap_x)
        sensemap_x = tf.expand_dims(sensemap_x, axis=0)
        sensemap_x = tf.image.resize_bicubic(sensemap_x, out_shape)
        sensemap_x = sensemap_x[0, :, :, :]
        sensemap_x = tfmri.channels_to_complex(sensemap_x)
    else:
        # Make sure size is correct
        map_shape = tf.shape(sensemap_x)
        map_shape_z = tf.slice(map_shape, [0], [1])
        map_shape_y = tf.slice(map_shape, [1], [1])
        assert_z = tf.assert_equal(out_shape[0], map_shape_z)
        assert_y = tf.assert_equal(out_shape[1], map_shape_y)
        with tf.control_dependencies([assert_z, assert_y]):
            sensemap_x = tf.identity(sensemap_x, name='sensemap_size_check')
        sensemap_x = tf.image.resize_image_with_crop_or_pad(
            sensemap_x, out_shape[0], out_shape[1])
    sensemap_x = tf.reshape(sensemap_x,
                            [out_shape[0], out_shape[1], num_maps, num_channels])

    if shuffle_channels:
        logger.info('  Shuffling channels...')
        with tf.variable_scope('shuffle_channels'):
            # place channel in first dimension and shuffle that
            sensemap_x = tf.reshape(
                sensemap_x, [out_shape[0], out_shape[1] * num_maps, num_channels])
            data_all = tf.concat([sensemap_x, ks_x], axis=1)

            def get_rot(num_channels):
                rot = ortho_group.rvs(num_channels)
                rot = np.expand_dims(rot.T, axis=0)
                rot = np.repeat(rot, out_shape[0], axis=0)
                rot = rot.astype(np.complex64)
                return rot
            rot = tf.py_func(get_rot, [num_channels], tf.complex64)
            data_all = tf.matmul(data_all, rot)

            index = 0
            sensemap_x = data_all[:, index:(
                index + num_maps * out_shape[1]), :]
            index += num_maps * out_shape[1]
            ks_x = data_all[:, index:(index + out_shape[1]), :]

            sensemap_x = tf.reshape(
                sensemap_x,
                [out_shape[0], out_shape[1], num_maps, num_channels])

    # Ground truth
    ks_truth = ks_x
    # Masked input
    ks_x = tf.multiply(ks_x, mask_x)

    features = {'xslice': tf.identity(xslice, name='xslixe'),
                'ks_input': ks_x,
                'sensemap': sensemap_x,
                'mask_recon': mask_recon,
                'scale': scale,
                'shape_c': shape_c}

    return features, ks_truth


def create_dataset(train_data_dir, mask_data_dir,
                   batch_size=16,
                   buffer_size=10,
                   out_shape=[80, 180],
                   init_shape=None,
                   shape_calib=20,
                   shape_scale=5,
                   scale_factor=1,
                   repeat=-1,
                   num_channels=6, num_maps=2,
                   bandpass_pad=0,
                   shuffle_channels=True,
                   random_seed=0,
                   name='create_dataset'):
    """Setups input tensors."""
    files = tf.data.Dataset.list_files(
        train_data_dir + '/*.tfrecords', shuffle=True)

    if mask_data_dir:
        mask_filenames_cfl = prepare_filenames(mask_data_dir,
                                               search_str='/*.cfl')
        masks = load_masks_cfl(mask_filenames_cfl)
    else:
        masks = None

    num_files = len(glob.glob(train_data_dir + '/*.tfrecords'))
    logger.info('Number of example files ({}): {}'.format(
        train_data_dir, num_files))
    if mask_data_dir:
        logger.info('Number of mask files ({}): {}'.format(
            mask_data_dir, len(mask_filenames_cfl)))

    with tf.variable_scope(name):
        dataset = files.apply(tf.data.experimental.parallel_interleave(
            tf.data.TFRecordDataset, cycle_length=batch_size*2))

        def _prep_tfrecord_with_param(example):
            return prep_tfrecord(example, masks, out_shape=out_shape,
                                 init_shape=init_shape, bandpass_pad=bandpass_pad,
                                 shape_calib=shape_calib,
                                 shape_scale=shape_scale,
                                 scale_factor=scale_factor,
                                 num_channels=num_channels, num_maps=num_maps,
                                 shuffle_channels=shuffle_channels,
                                 resize_sensemaps=True,
                                 random_seed=random_seed)
        dataset = dataset.map(_prep_tfrecord_with_param,
                              num_parallel_calls=6)
        dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(
            batch_size * buffer_size, count=repeat, seed=random_seed))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(6)

    return dataset
