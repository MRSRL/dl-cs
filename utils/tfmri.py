"""Common functions for setup."""
import tensorflow as tf
import numpy as np
import scipy.signal


def complex_to_channels(
        image,
        data_format='channels_last',
        name='complex2channels'):
    """Convert data from complex to channels."""
    if len(image.shape) != 3 and len(image.shape) != 4:
        raise TypeError('Input data must be have 3 or 4 dimensions')

    axis_c = -1 if data_format == 'channels_last' else -3

    if image.dtype is not tf.complex64 and image.dtype is not tf.complex128:
        raise TypeError('Input data must be complex')

    with tf.name_scope(name):
        image_out = tf.concat((tf.real(image), tf.imag(image)), axis_c)
    return image_out


def channels_to_complex(
        image,
        data_format='channels_last',
        name='channels2complex'):
    """Convert data from channels to complex."""
    if len(image.shape) != 3 and len(image.shape) != 4:
        raise TypeError('Input data must be have 3 or 4 dimensions')

    axis_c = -1 if data_format == 'channels_last' else -3
    shape_c = image.shape[axis_c].value

    if shape_c and (shape_c % 2 != 0):
        raise TypeError(
            'Number of channels (%d) must be divisible by 2' %
            shape_c)
    if image.dtype is tf.complex64 or image.dtype is tf.complex128:
        raise TypeError('Input data cannot be complex')

    with tf.name_scope(name):
        image_real, image_imag = tf.split(image, 2, axis=axis_c)
        image_out = tf.complex(image_real, image_imag)
    return image_out


def circular_pad(tf_input, pad, axis):
    """Perform circular padding."""
    shape_input = tf.shape(tf_input)
    shape_0 = tf.cast(tf.reduce_prod(shape_input[:axis]), dtype=tf.int32)
    shape_axis = shape_input[axis]
    tf_output = tf.reshape(tf_input, tf.stack((shape_0, shape_axis, -1)))

    tf_pre = tf_output[:, shape_axis - pad:, :]
    tf_post = tf_output[:, :pad, :]
    tf_output = tf.concat((tf_pre, tf_output, tf_post), axis=1)

    shape_out = tf.concat((shape_input[:axis],
                           [shape_axis + 2 * pad],
                           shape_input[axis + 1:]), axis=0)
    tf_output = tf.reshape(tf_output, shape_out)

    return tf_output


def fftshift(im, axis=0, name='fftshift'):
    """Perform fft shift.

    This function assumes that the axis to perform fftshift is divisible by 2.

    Args:
        axis (int, or array of ints): Axes to perform shift operation.
        name (str): TensorFlow name scope.

    Returns:
        Tensor with the contents fft shifted.
    """
    with tf.name_scope(name):
        if not hasattr(axis, '__iter__'):
            axis = [axis]
        output = im
        for a in axis:
            split0, split1 = tf.split(output, 2, axis=a)
            output = tf.concat((split1, split0), axis=a)

    return output


def fftc(
        im,
        data_format='channels_last',
        orthonorm=True,
        transpose=False,
        name='fftc'):
    """Centered FFT on last non-channel dimension."""
    with tf.name_scope(name):
        im_out = im
        if data_format == 'channels_last':
            permute_orig = np.arange(len(im.shape))
            permute = permute_orig.copy()
            permute[-2] = permute_orig[-1]
            permute[-1] = permute_orig[-2]
            im_out = tf.transpose(im_out, permute)

        if orthonorm:
            fftscale = tf.sqrt(tf.cast(im_out.shape[-1], tf.float32))
        else:
            fftscale = 1.0
        fftscale = tf.cast(fftscale, dtype=tf.complex64)

        im_out = fftshift(im_out, axis=-1)
        if transpose:
            im_out = tf.ifft(im_out) * fftscale
        else:
            im_out = tf.fft(im_out) / fftscale
        im_out = fftshift(im_out, axis=-1)

        if data_format == 'channels_last':
            im_out = tf.transpose(im_out, permute)

    return im_out


def ifftc(im, data_format='channels_last', orthonorm=True, name='ifftc'):
    """Centered IFFT on last non-channel dimension."""
    return fftc(
        im,
        data_format=data_format,
        orthonorm=orthonorm,
        transpose=True,
        name=name)


def fft2c(
        im,
        data_format='channels_last',
        orthonorm=True,
        transpose=False,
        name='fft2c'):
    """Centered FFT2 on last two non-channel dimensions."""
    with tf.name_scope(name):
        im_out = im
        if data_format == 'channels_last':
            permute_orig = np.arange(len(im.shape))
            permute = permute_orig.copy()
            permute[-3] = permute_orig[-1]
            permute[-2:] = permute_orig[-3:-1]
            im_out = tf.transpose(im_out, permute)

        if orthonorm:
            fftscale = tf.sqrt(tf.cast(im_out.shape[-1], tf.float32)
                               * tf.cast(im_out.shape[-2], tf.float32))
        else:
            fftscale = 1.0
        fftscale = tf.cast(fftscale, dtype=tf.complex64)

        im_out = fftshift(im_out, axis=(-2, -1))
        if transpose:
            im_out = tf.ifft2d(im_out) * fftscale
        else:
            im_out = tf.fft2d(im_out) / fftscale
        im_out = fftshift(im_out, axis=(-2, -1))

        if data_format == 'channels_last':
            permute[-3:-1] = permute_orig[-2:]
            permute[-1] = permute_orig[-3]
            im_out = tf.transpose(im_out, permute)

    return im_out


def ifft2c(im, data_format='channels_last', orthonorm=True, name='ifft2c'):
    """Centered IFFT2 on last two non-channel dimensions."""
    return fft2c(
        im,
        data_format=data_format,
        orthonorm=orthonorm,
        transpose=True,
        name=name)


def sumofsq(image_in, keep_dims=False, axis=-1, name='sumofsq'):
    """Compute square root of sum of squares."""
    with tf.variable_scope(name):
        image_out = tf.square(tf.abs(image_in))
        image_out = tf.reduce_sum(image_out, keepdims=keep_dims,
                                  axis=axis)
        image_out = tf.sqrt(image_out)

    return image_out


def conj_kspace(image_in, name='kspace_conj'):
    """Conjugate k-space data."""
    with tf.variable_scope(name):
        image_out = tf.reverse(image_in, axis=[1])
        image_out = tf.reverse(image_out, axis=[2])
        mod = np.zeros((1, 1, 1, image_in.get_shape().as_list()[-1]))
        mod[:, :, :, 1::2] = -1
        mod = tf.constant(mod, dtype=tf.float32)
        image_out = tf.multiply(image_out, mod)

    return image_out


def replace_kspace(image_orig, image_cur, name='replace_kspace'):
    """Replace k-space with known values."""
    with tf.variable_scope(name):
        mask_x = kspace_mask(image_orig)
        image_out = tf.add(tf.multiply(mask_x, image_orig),
                           tf.multiply((1 - mask_x), image_cur))

    return image_out


def kspace_mask(image_orig, name='kspace_mask', dtype=None):
    """Find k-space mask."""
    with tf.variable_scope(name):
        mask_x = tf.not_equal(image_orig, 0)
        if dtype is not None:
            mask_x = tf.cast(mask_x, dtype=dtype)
    return mask_x


def kspace_threshhold(image_orig, threshhold=1e-8, name='kspace_threshhold'):
    """Find k-space mask based on threshhold.

    Anything less the specified threshhold is set to 0.
    Anything above the specified threshhold is set to 1.
    """
    with tf.variable_scope(name):
        mask_x = tf.greater(tf.abs(image_orig), threshhold)
        mask_x = tf.cast(mask_x, dtype=tf.float32)
    return mask_x


def kspace_location(image_size):
    """Construct matrix with k-space normalized location."""
    x = np.arange(image_size[0], dtype=np.float32) / image_size[0] - 0.5
    y = np.arange(image_size[1], dtype=np.float32) / image_size[1] - 0.5
    xg, yg = np.meshgrid(x, y)
    out = np.stack((xg.T, yg.T))
    return out


def tf_kspace_location(tf_shape_y, tf_shape_x):
    """Construct matrix with k-psace normalized location as tensor."""
    tf_y = tf.cast(tf.range(tf_shape_y), tf.float32)
    tf_y = tf_y / tf.cast(tf_shape_y, tf.float32) - 0.5
    tf_x = tf.cast(tf.range(tf_shape_x), tf.float32)
    tf_x = tf_x / tf.cast(tf_shape_x, tf.float32) - 0.5

    [tf_yg, tf_xg] = tf.meshgrid(tf_y, tf_x)
    tf_yg = tf.transpose(tf_yg, [1, 0])
    tf_xg = tf.transpose(tf_xg, [1, 0])
    out = tf.stack((tf_yg, tf_xg))
    return out


def kspace_radius(image_size):
    """Construct matrix with k-space radius."""
    x = np.arange(image_size[0], dtype=np.float32) / image_size[0] - 0.5
    y = np.arange(image_size[1], dtype=np.float32) / image_size[1] - 0.5
    xg, yg = np.meshgrid(x, y)
    kr = np.sqrt(xg * xg + yg * yg)
    return kr.T


def sensemap_model(x, sensemap, transpose=False,
                   data_format='channels_last', name='sensemap_model'):
    """Apply sensitivity maps.

    Args
       x: data input [(batch), height, width, channels] for channels_last
       sensemap: sensitivity maps [(batch), height, width, maps, coils]
       tranpose: boolean to specify forward or transpose model
       data_format: 'channels_last' or 'channels_first'
    """
    if data_format == 'channels_last':
        # [batch, height, width, maps, coils]
        axis_m, axis_c = -2, -1
    else:
        # [batch, maps, coils, height, width]
        axis_m, axis_c = -4, -3
    with tf.name_scope(name):
        if transpose:
            x_shape = x.get_shape().as_list()
            x = tf.expand_dims(x, axis=axis_m)
            x = tf.multiply(tf.conj(sensemap), x)
            x = tf.reduce_sum(x, axis=axis_c)
        else:
            x = tf.expand_dims(x, axis=axis_c)
            x = tf.multiply(x, sensemap)
            x = tf.reduce_sum(x, axis=axis_m)
    return x


def model_forward(
        x,
        sensemap,
        data_format='channels_last',
        name='model_forward'):
    """Apply forward model.

    Image domain to k-space domain.
    """
    with tf.name_scope(name):
        if sensemap is not None:
            x = sensemap_model(
                x,
                sensemap,
                transpose=False,
                data_format=data_format)
        x = fft2c(x, data_format=data_format)
    return x


def model_transpose(
        x,
        sensemap,
        data_format='channels_last',
        name='model_transpose'):
    """Apply transpose model.

    k-Space domain to image domain
    """
    with tf.name_scope(name):
        x = ifft2c(x, data_format=data_format)
        if sensemap is not None:
            x = sensemap_model(
                x,
                sensemap,
                transpose=True,
                data_format=data_format)
    return x
