"""Runs model on data input"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import model
import data
import argparse
import sigpy.mri
from tqdm import tqdm
from matplotlib import pyplot
from utils import mri
from utils import fftc
from utils import cfl
import utils.logging

logger = utils.logging.logger


class DeepRecon:
    def __init__(self,
                 model_dir,
                 num_channels,
                 shape_z,
                 shape_y,
                 shape_scale=5,
                 num_maps=1,
                 batch_size=1,
                 tf_graph=None,
                 tf_sess=None,
                 log_level=utils.logging.logging.WARNING,
                 debug_plot=False):
        """
        Setup model for inference

        Args:
          model_dir: Directory with model files
          num_channels: Number of channels for input data
          shape_z: Shape of input data in Z
          shape_y: Shape of input data in Y
          shape_scale: Scale data with center k-space data
          num_maps: Number of sets of sensitivity maps
        """
        self.debug_plot = debug_plot

        self.tf_graph = tf_graph
        if self.tf_graph is None:
            self.tf_graph = tf.Graph()
        self.tf_sess = tf_sess
        if self.tf_sess is None:
            session_config = tf.ConfigProto()
            session_config.gpu_options.allow_growth = True  # pylint: disable=E1101
            session_config.allow_soft_placement = True
            self.tf_sess = tf.Session(
                graph=self.tf_graph, config=session_config)

        params = model.load_params(model_dir)

        with self.tf_graph.as_default():
            self.batch_size = batch_size
            self.tf_kspace_input = tf.placeholder(
                tf.complex64,
                (self.batch_size, shape_z, shape_y, num_channels))
            self.tf_sensemap_input = tf.placeholder(
                tf.complex64,
                (self.batch_size, shape_z, shape_y, num_maps, num_channels))

            if shape_scale > 0:
                scale = tf.image.resize_image_with_crop_or_pad(
                    self.tf_kspace_input, shape_scale, shape_scale)
                scale = tf.reduce_mean(tf.square(tf.abs(scale)))
                scale *= shape_scale * shape_scale / shape_y / shape_z
            else:
                logger.info('Turning off scaling...')
                scale = 1.0
            scale = tf.cast(1.0 / tf.sqrt(scale), dtype=tf.complex64)
            tf_kspace_input_scaled = self.tf_kspace_input * scale
            tf_image_output_scaled, tf_kspace_output_scaled, self.iter_out = model.unrolled_prox(
                tf_kspace_input_scaled,
                self.tf_sensemap_input,
                num_grad_steps=params['unrolled_steps'],
                resblock_num_features=params['unrolled_num_features'],
                resblock_num_blocks=params['unrolled_num_resblocks'],
                resblock_share=params['unrolled_share'],
                training=False,
                hard_projection=params['hard_projection'],
                scope=params.get('recon_scope', 'ReconNetwork'))
            self.tf_image_output = tf_image_output_scaled / scale
            self.tf_kspace_output = tf_kspace_output_scaled / scale

            filename_latest_model = tf.train.latest_checkpoint(model_dir)
            logger.info('Loading model ({})...'.format(filename_latest_model))
            saver = tf.train.Saver()
            saver.restore(self.tf_sess, filename_latest_model)

    def run(self, kspace, sensemap):
        """
        Run inference on dataset

        Args
          kspace: (channels, kz, ky, x)
          sensemap: (maps, channels, z, y, x)
        """
        logger.info('IFFT in x...')
        kspace_input = fftc.ifftc(kspace, axis=-1)
        # (channels, kz, ky, x) to (x, kz, ky, channels)
        kspace_input = np.transpose(kspace_input, (3, 1, 2, 0))
        kspace_output = np.zeros(kspace_input.shape, dtype=np.complex64)

        if self.debug_plot:
            image_input = fftc.ifftc(fftc.ifftc(kspace_input, axis=1), axis=2)
            image_input = mri.sumofsq(image_input, axis=-1)
            image_output = np.zeros(image_input.shape, dtype=np.float64)

        # tranpose to (x, kz, ky, maps, channels)
        sensemap_input = np.transpose(sensemap, (4, 2, 3, 0, 1))
        num_x = kspace_input.shape[0]
        num_batches = int(np.ceil(1.0 * num_x / self.batch_size))

        logger.info('Running inference ({} batches)...'.format(num_batches))

        def wrap(x):
            return x

        if logger.getEffectiveLevel() is utils.logging.logging.INFO:
            wrap = tqdm
        for b in wrap(range(num_batches)):
            x_start = b * self.batch_size
            x_end = (b + 1) * self.batch_size
            logger.debug('  batch {}/{}: ({}, {})'.format(
                b, num_batches, x_start, x_end))
            kspace_input_batch = kspace_input[x_start:x_end, :, :, :].copy()
            sensemap_input_batch = sensemap_input[x_start:x_end, :, :, :]
            x_act_end = kspace_input_batch.shape[0] + x_start
            if x_end != x_act_end:
                pad = x_end - x_act_end
                kspace_input_batch = np.concatenate(
                    (kspace_input_batch,
                     np.zeros((pad, ) + kspace_input_batch.shape[1:],
                              np.complex64)),
                    axis=0)
                sensemap_input_batch = np.concatenate(
                    (sensemap_input_batch,
                     np.zeros((pad, ) + sensemap_input_batch.shape[1:],
                              np.complex64)),
                    axis=0)
            feed_dict = {
                self.tf_kspace_input: kspace_input_batch,
                self.tf_sensemap_input: sensemap_input_batch
            }
            out = self.tf_sess.run([self.tf_kspace_output],
                                   feed_dict=feed_dict)[0]
            kspace_output[x_start:x_act_end, :, :, :] = out

            if self.debug_plot:
                imout = fftc.ifftc(fftc.ifftc(out, axis=1), axis=2)
                imout = mri.sumofsq(imout, axis=-1)
                image_output[x_start:x_act_end, :, :] = imout

                image_axial_disp = np.concatenate(
                    (image_input[x_start, :, :], image_output[x_start, :, :]),
                    axis=1)
                image_sag_disp = np.concatenate(
                    (image_input[:, :, image_input.shape[-1] // 2],
                     image_output[:, :, image_output.shape[-1] // 2]),
                    axis=1)
                pyplot.figure(1)
                pyplot.subplot(2, 1, 1)
                pyplot.imshow(image_axial_disp, cmap='gray')
                pyplot.axis('off')
                pyplot.title('Processed: {}/{}'.format(b, num_batches))
                pyplot.subplot(2, 1, 2)
                pyplot.imshow(image_sag_disp, cmap='gray')
                pyplot.axis('off')
                pyplot.pause(0.01)

        # (x, kz, ky, channels) to (channels, kz, ky, x)
        kspace_output = np.transpose(kspace_output, (3, 1, 2, 0))

        logger.info('FFT in x...')
        kspace_output = fftc.fftc(kspace_output, axis=-1)

        return kspace_output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference')
    parser.add_argument(
        'model_dir', action='store', help='Location of trained model')
    parser.add_argument(
        'kspace_input', action='store', help='CFL file of kspace input data')
    parser.add_argument(
        'kspace_output', action='store', help='CFL file of kspace output data')
    parser.add_argument(
        '--sensemap', default=None, help='Insert sensemap as CFL')
    parser.add_argument('--device', default='0', help='GPU device to use')
    parser.add_argument(
        '--batch_size', default=1, type=int, help='Batch size for inference')
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose printing (default: False)')
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Plotting for debugging (default: False)')
    args = parser.parse_args()

    log_level = utils.logging.logging.INFO if args.verbose else utils.logging.logging.WARNING
    logger.setLevel(log_level)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    logger.info('Using GPU device {}...'.format(args.device))

    logger.info('Loading k-space data from {}...'.format(args.kspace_input))
    kspace = np.load(args.kspace_input)
    sensemap = None
    if args.sensemap and os.path.isfile(args.sensemap):
        logger.info('Loading sensitivity maps from {}...'.format(
            args.sensemap))
        sensemap = np.load(args.sensemap)
    else:
        logger.info('Estimating sensitivity maps...')
        sensemap = mri.estimate_sense_maps(kspace)
        if args.sensemap:
            logger.info('  Saving sensitivity maps to {}...'.format(
                args.sensemap))
            np.save(args.sensemap, sensemap)
    sensemap = np.squeeze(sensemap)
    if sensemap.ndim != 5:
        # (maps, channels, z, y, x)
        sensemap = np.expand_dims(sensemap, axis=0)

    logger.info('Setting up model from {}...'.format(args.model_dir))
    num_channels = kspace.shape[0]
    shape_z = kspace.shape[1]
    shape_y = kspace.shape[2]
    model = DeepRecon(
        args.model_dir,
        num_channels,
        shape_z,
        shape_y,
        batch_size=args.batch_size,
        log_level=log_level,
        debug_plot=args.plot)

    logger.info('Running inference...')
    kspace_output = model.run(kspace, sensemap)

    logger.info('Writing output to {}...'.format(args.kspace_output))
    cfl.write(args.kspace_output, kspace_output)

    logger.info('Finished')
