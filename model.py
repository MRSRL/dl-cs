"""MRI model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from utils import tfmri
import common

logger = common.logger


def _batch_norm(tf_input, data_format='channels_last', training=False):
    axis_c = -1 if data_format == 'channels_last' else 1
    tf_output = tf.layers.batch_normalization(
        tf_input, axis=axis_c, training=training, fused=True)
    # tf_output = tf.keras.layers.BatchNormalization(axis=axis_c)(
    #    tf_input, training=training)
    return tf_output


def _batch_norm_relu(tf_input, data_format='channels_last',
                     batchnorm=True, training=False):
    if batchnorm:
        tf_output = _batch_norm(
            tf_input, data_format=data_format, training=training)
    else:
        tf_output = tf_input
    tf_output = tf.nn.relu(tf_output)
    return tf_output


def _conv2d(tf_input, num_features=128, kernel_size=3, data_format='channels_last',
            circular=True, use_bias=False):
    """Conv2d with option for circular convolution."""
    if data_format == 'channels_last':
        # (batch, z, y, channels)
        axis_z = 1
        axis_y = 2
    else:
        # (batch, channels, z, y)
        axis_z = 2
        axis_y = 3

    tf_output = tf_input
    shape_input = tf.shape(tf_input)
    shape_z = shape_input[axis_z]
    shape_y = shape_input[axis_y]
    pad = int((kernel_size - 0.5) / 2)

    if circular and pad > 0:
        with tf.name_scope('circular_pad'):
            tf_output = tfmri.circular_pad(tf_output, pad, axis_z)
            tf_output = tfmri.circular_pad(tf_output, pad, axis_y)

    tf_output = tf.layers.conv2d(tf_output, num_features, kernel_size,
                                 padding='same', use_bias=use_bias,
                                 data_format=data_format)
    # tf_output = tf.keras.layers.Conv2D(num_features, kernel_size,
    #                                   padding='same', use_bias=use_bias,
    #                                   data_format=data_format)(tf_output)

    if circular and pad > 0:
        with tf.name_scope('circular_crop'):
            if data_format == 'channels_last':
                tf_output = tf_output[:, pad:(shape_z + pad),
                                      pad:(shape_y + pad), :]
            else:
                tf_output = tf_output[:, :, pad:(shape_z + pad),
                                      pad:(shape_y + pad)]

    return tf_output


def _res_block(net_input, num_features=32, kernel_size=3,
               data_format='channels_last', circular=True,
               batchnorm=True, training=True, name='res_block'):
    """Create ResNet block.

    [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        Identity Mappings in Deep Residual Networks. arXiv: 1603.05027
    """
    if data_format == 'channels_last':
        axis_z = 1
        axis_y = 2
        axis_c = 3
    else:
        axis_c = 1
        axis_z = 2
        axis_y = 3
    shape_z = tf.shape(net_input)[axis_z]
    shape_y = tf.shape(net_input)[axis_y]
    shape_c = net_input.shape[axis_c]
    pad = int((2 * (kernel_size - 1) + 0.5) / 2)

    with tf.name_scope(name):

        shortcut = net_input
        if num_features != shape_c:
            shortcut = _conv2d(shortcut, num_features=num_features, kernel_size=1,
                               data_format=data_format, circular=False,
                               use_bias=(not batchnorm))

        net_cur = net_input

        if circular:
            with tf.name_scope('circular_pad'):
                net_cur = tfmri.circular_pad(net_cur, pad, axis_z)
                net_cur = tfmri.circular_pad(net_cur, pad, axis_y)

        net_cur = _batch_norm_relu(net_cur, data_format=data_format,
                                   batchnorm=batchnorm,
                                   training=training)
        net_cur = _conv2d(net_cur, num_features=num_features, kernel_size=kernel_size,
                          data_format=data_format, circular=False, use_bias=(not batchnorm))
        net_cur = _batch_norm_relu(net_cur, data_format=data_format,
                                   batchnorm=batchnorm,
                                   training=training)
        net_cur = _conv2d(net_cur, num_features=num_features, kernel_size=kernel_size,
                          data_format=data_format, circular=False, use_bias=(not batchnorm))

        if circular:
            with tf.name_scope('circular_crop'):
                if data_format == 'channels_last':
                    net_cur = net_cur[:, pad:(pad + shape_z),
                                      pad:(pad + shape_y), :]
                else:
                    net_cur = net_cur[:, :, pad:(pad + shape_z),
                                      pad:(pad + shape_y)]

        net_cur = net_cur + shortcut

    return net_cur


def prox_res_net(curr_x,
                 num_features=32, kernel_size=3,
                 num_blocks=3,
                 circular=True,
                 data_format='channels_last',
                 do_residual=True,
                 batchnorm=True,
                 training=True, num_features_out=None,
                 name='prox_res_net'):
    """Create prior gradient."""
    if data_format == 'channels_last':
        axis_z = 1
        axis_y = 2
        axis_c = 3
    else:
        axis_c = 1
        axis_z = 2
        axis_y = 3

    num_features_in = curr_x.shape[axis_c]
    if num_features_out is None:
        num_features_out = num_features_in

    shape_z = tf.shape(curr_x)[axis_z]
    shape_y = tf.shape(curr_x)[axis_y]
    num_conv2d = num_blocks * 2 + 1
    pad = int((num_conv2d * (kernel_size - 1) + 0.5) / 2)

    with tf.name_scope(name):
        net = curr_x
        shortcut = net

        if do_residual and (num_features_in != num_features_out):
            shortcut = _conv2d(shortcut, num_features=num_features_out, kernel_size=1,
                               data_format=data_format, circular=False,
                               use_bias=(not batchnorm))

        if circular:
            with tf.name_scope('circular_pad'):
                net = tfmri.circular_pad(net, pad, axis_z)
                net = tfmri.circular_pad(net, pad, axis_y)

        for _ in range(num_blocks):
            net = _res_block(net, training=training, num_features=num_features,
                             kernel_size=kernel_size, data_format=data_format,
                             batchnorm=batchnorm, circular=False)

        # Save network before last conv for densely connected network
        net_dense = net

        net = _batch_norm_relu(net, data_format=data_format,
                               batchnorm=batchnorm, training=training)
        net = _conv2d(net, num_features=num_features_out, kernel_size=kernel_size,
                      data_format=data_format, circular=False,
                      use_bias=(not batchnorm))

        if circular:
            with tf.name_scope('circular_crop'):
                if data_format == 'channels_last':
                    net = net[:, pad:(pad + shape_z), pad:(pad + shape_y), :]
                    net_dense = net_dense[:, pad:(
                        pad + shape_z), pad:(pad + shape_y), :]
                else:
                    net = net[:, :, pad:(pad + shape_z), pad:(pad + shape_y)]
                    net_dense = net_dense[:, :, pad:(
                        pad + shape_z), pad:(pad + shape_y)]

        if do_residual:
            net = net + shortcut

    return net, net_dense


def unrolled_prox(ks_input, sensemap,
                  num_grad_steps=4,
                  resblock_num_features=128,
                  resblock_num_blocks=3,
                  resblock_share=False,
                  training=True,
                  mask_output=1,
                  hard_projection=True,
                  do_dense=False,
                  batchnorm=True,
                  circular=True,
                  fix_update=False,
                  mask=None,
                  scope='UnrolledProx'):
    """Create general unrolled network for MRI.

    We are trying to solve the optimization
        \hat{x} = \| A x - b \|_2^2
    with a learned proximal operator.

    x_{k+1} = prox( x_k - 2 * t * A^T (A x- b) )
            = prox( x_k - 2 * t * (A^T A x - A^T b))
    """
    summary_iter = {}

    logger.info('Building unrolled network....')
    logger.info('  Num of gradient steps: {}'.format(num_grad_steps))
    logger.info('  Prior: {} ResBlocks, {} features'.format(
        resblock_num_blocks, resblock_num_features))
    if resblock_share:
        logger.info('  Sharing weights...')
    if sensemap is not None:
        logger.info('  Using sensitivity maps...')
    if do_dense:
        logger.info('  Inserting dense connections...')
    if not batchnorm:
        logger.info('  Turning off batch normalization...')
    if not circular:
        logger.info('  Warning! No circular convolutions...')

    with tf.variable_scope(scope):
        ks_input = tf.identity(ks_input, name='input_kspace')
        sensemap = tf.identity(sensemap, name='input_sensemap')
        mask = tfmri.kspace_mask(ks_input, dtype=tf.complex64)

        ks_0 = ks_input
        # x0 = A^T W b
        im_0 = tfmri.model_transpose(ks_0, sensemap)
        im_0 = tf.identity(im_0, name='input_image')

        # To be updated
        ks_k = ks_0
        im_k = im_0
        im_dense = None

        for i_step in range(num_grad_steps):
            iter_name = 'iter_%02d' % i_step
            if resblock_share:
                scope_name = 'iter'
            else:
                scope_name = iter_name

            with tf.variable_scope(scope_name,
                                   reuse=(tf.AUTO_REUSE if resblock_share else False)):
                # = S( x_k - 2 * t * (A^T W A x_k - A^T W b))
                # = S( x_k - 2 * t * (A^T W A x_k - x0))
                with tf.variable_scope('update'):
                    im_k_orig = im_k

                    # xk = A^T A x_k
                    ks_k = tfmri.model_forward(im_k, sensemap)
                    ks_k = mask * ks_k
                    im_k = tfmri.model_transpose(ks_k, sensemap)
                    # xk = A^T A x_k - A^T b
                    im_k = tfmri.complex_to_channels(im_k - im_0)
                    im_k_orig = tfmri.complex_to_channels(im_k_orig)
                    if fix_update:
                        t_update = -2.0
                    else:
                        t_update = tf.get_variable('t', dtype=tf.float32,
                                                   initializer=tf.constant([-2.0]))
                    im_k = im_k_orig + t_update * im_k

                with tf.variable_scope('prox'):
                    num_channels_out = im_k.shape[-1]
                    # Default is channels last
                    # Change to channels_first
                    im_k = tf.transpose(im_k, [0, 3, 1, 2])
                    if im_dense is not None:
                        im_k = tf.concat([im_k, im_dense], axis=1)
                    im_k, im_dense_k = prox_res_net(
                        im_k, training=training,
                        num_features=resblock_num_features,
                        num_blocks=resblock_num_blocks,
                        circular=circular,
                        num_features_out=num_channels_out,
                        data_format='channels_first',
                        batchnorm=batchnorm)
                    if do_dense:
                        if im_dense is not None:
                            im_dense = tf.concat(
                                [im_dense, im_dense_k], axis=1)
                        else:
                            im_dense = im_dense_k
                    im_k = tf.transpose(im_k, [0, 2, 3, 1])
                    im_k = tfmri.channels_to_complex(im_k)

                im_k = tf.identity(im_k, name='image')

                with tf.name_scope('summary'):
                    summary_iter[iter_name] = im_k

        ks_k = tfmri.model_forward(im_k, sensemap)
        if hard_projection:
            logger.info('   Final hard data projection...')
            ks_k = mask * ks_0 + (1 - mask) * ks_k
            if mask_output is not None:
                ks_k = ks_k * mask_output
            im_k = tfmri.model_transpose(ks_k, sensemap)
        else:
            if mask_output is not None:
                ks_k = ks_k * mask_output
                im_k = tfmri.model_transpose(ks_k, sensemap)

        ks_k = tf.identity(ks_k, name='output_kspace')
        im_k = tf.identity(im_k, name='output_image')

    return im_k, ks_k, summary_iter


def adversarial(x, num_features=32, num_blocks=3, kernel_size=3,
                batchnorm=True, data_format='channels_last',
                training=False, scope='Adversarial'):
    """Adversarial loss model

    Simple construction of adversarial loss using ResBlocks
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        x = tfmri.complex_to_channels(x, data_format=data_format)
        # channels last -> channels first
        if data_format is not 'channels_first':
            x = tf.transpose(x, [0, 3, 1, 2])

        data_format_b = 'channels_first'
        num_features_b = num_features
        for _ in range(num_blocks):
            x = _res_block(
                x, training=training, num_features=num_features_b, kernel_size=kernel_size,
                data_format=data_format_b, circular=True, batchnorm=batchnorm)
            # 1x1 convolutions with strides to reduce image size and increase features
            num_features_b *= 2
            x = _batch_norm_relu(
                x, data_format=data_format_b, batchnorm=batchnorm, training=training)
            x = tf.layers.conv2d(
                x, num_features_b, 1, padding='same', use_bias=(not batchnorm),
                strides=(2, 2), data_format=data_format_b)
            # x = tf.keras.layers.Conv2D(num_features_b, 1, strides=2,
            #                           padding='same', use_bias=(not batchnorm),
            #                           data_format=data_format_b)(x)
        if batchnorm:
            x = _batch_norm(x, data_format=data_format_b, training=training)
        x = tf.nn.tanh(x)
        if data_format is not 'channels_first':
            x = tf.transpose(x, [0, 2, 3, 1])
        x = tfmri.channels_to_complex(x, data_format=data_format)

    return x
