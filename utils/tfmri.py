"""Common functions for setup."""
import tensorflow as tf
import numpy as np
import scipy.signal


def complex_to_channels(image, name='complex2channels'):
    """Convert data from complex to channels."""
    with tf.name_scope(name):
        image_out = tf.stack([tf.real(image), tf.imag(image)], axis=-1)
        # tf.shape: returns tensor
        # image.shape: returns actual values
        shape_out = tf.concat([tf.shape(image)[:-1], [image.shape[-1]*2]],
                              axis=0)
        image_out = tf.reshape(image_out, shape_out)
    return image_out


def channels_to_complex(image, name='channels2complex'):
    """Convert data from channels to complex."""
    with tf.name_scope(name):
        image_out = tf.reshape(image, [-1, 2])
        image_out = tf.complex(image_out[:, 0], image_out[:, 1])
        shape_out = tf.concat([tf.shape(image)[:-1], [image.shape[-1] // 2]],
                              axis=0)
        image_out = tf.reshape(image_out, shape_out)
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
    """
    with tf.name_scope(name):
        split0, split1 = tf.split(im, 2, axis=axis)
        output = tf.concat((split1, split0), axis=axis)

    return output


def ifftc(im, name='ifftc', do_orthonorm=True):
    """Centered iFFT on second to last dimension."""
    with tf.name_scope(name):
        im_out = im
        if do_orthonorm:
            fftscale = tf.sqrt(1.0 * im_out.get_shape().as_list()[-2])
        else:
            fftscale = 1.0
        fftscale = tf.cast(fftscale, dtype=tf.complex64)
        if len(im.get_shape()) == 4:
            im_out = tf.transpose(im_out, [0, 3, 1, 2])
            im_out = fftshift(im_out, axis=3)
        else:
            im_out = tf.transpose(im_out, [2, 0, 1])
            im_out = fftshift(im_out, axis=2)
        with tf.device('/gpu:0'):
            # FFT is only supported on the GPU
            im_out = tf.ifft(im_out) * fftscale
        if len(im.get_shape()) == 4:
            im_out = fftshift(im_out, axis=3)
            im_out = tf.transpose(im_out, [0, 2, 3, 1])
        else:
            im_out = fftshift(im_out, axis=2)
            im_out = tf.transpose(im_out, [1, 2, 0])

    return im_out


def fftc(im, name='fftc', do_orthonorm=True):
    """Centered FFT on second to last dimension."""
    with tf.name_scope(name):
        im_out = im
        if do_orthonorm:
            fftscale = tf.sqrt(1.0 * im_out.get_shape().as_list()[-2])
        else:
            fftscale = 1.0
        fftscale = tf.cast(fftscale, dtype=tf.complex64)
        if len(im.get_shape()) == 4:
            im_out = tf.transpose(im_out, [0, 3, 1, 2])
            im_out = fftshift(im_out, axis=3)
        else:
            im_out = tf.transpose(im_out, [2, 0, 1])
            im_out = fftshift(im_out, axis=2)
        with tf.device('/gpu:0'):
            im_out = tf.fft(im_out) / fftscale
        if len(im.get_shape()) == 4:
            im_out = fftshift(im_out, axis=3)
            im_out = tf.transpose(im_out, [0, 2, 3, 1])
        else:
            im_out = fftshift(im_out, axis=2)
            im_out = tf.transpose(im_out, [1, 2, 0])

    return im_out


def ifft2c(im, name='ifft2c', do_orthonorm=True):
    """Centered iFFT2."""
    with tf.name_scope(name):
        im_out = im
        if do_orthonorm:
            fftscale = tf.sqrt(1.0 * im_out.get_shape().as_list()[-2]
                               * im_out.get_shape().as_list()[-3])
        else:
            fftscale = 1.0
        fftscale = tf.cast(fftscale, dtype=tf.complex64)
        if len(im.get_shape()) == 5:
            im_out = tf.transpose(im_out, [0, 3, 4, 1, 2])
            im_out = fftshift(im_out, axis=4)
            im_out = fftshift(im_out, axis=3)
        elif len(im.get_shape()) == 4:
            im_out = tf.transpose(im_out, [0, 3, 1, 2])
            im_out = fftshift(im_out, axis=3)
            im_out = fftshift(im_out, axis=2)
        else:
            im_out = tf.transpose(im_out, [2, 0, 1])
            im_out = fftshift(im_out, axis=2)
            im_out = fftshift(im_out, axis=1)

        with tf.device('/gpu:0'):
            # FFT is only supported on the GPU
            im_out = tf.ifft2d(im_out) * fftscale

        if len(im.get_shape()) == 5:
            im_out = fftshift(im_out, axis=4)
            im_out = fftshift(im_out, axis=3)
            im_out = tf.transpose(im_out, [0, 3, 4, 1, 2])
        elif len(im.get_shape()) == 4:
            im_out = fftshift(im_out, axis=3)
            im_out = fftshift(im_out, axis=2)
            im_out = tf.transpose(im_out, [0, 2, 3, 1])
        else:
            im_out = fftshift(im_out, axis=2)
            im_out = fftshift(im_out, axis=1)
            im_out = tf.transpose(im_out, [1, 2, 0])

    return im_out


def fft2c(im, name='fft2c', do_orthonorm=True):
    """Centered FFT2."""
    with tf.name_scope(name):
        im_out = im
        if do_orthonorm:
            fftscale = tf.sqrt(1.0 * im_out.get_shape().as_list()[-2]
                               * im_out.get_shape().as_list()[-3])
        else:
            fftscale = 1.0
        fftscale = tf.cast(fftscale, dtype=tf.complex64)
        if len(im.get_shape()) == 5:
            im_out = tf.transpose(im_out, [0, 3, 4, 1, 2])
            im_out = fftshift(im_out, axis=4)
            im_out = fftshift(im_out, axis=3)
        elif len(im.get_shape()) == 4:
            im_out = tf.transpose(im_out, [0, 3, 1, 2])
            im_out = fftshift(im_out, axis=3)
            im_out = fftshift(im_out, axis=2)
        else:
            im_out = tf.transpose(im_out, [2, 0, 1])
            im_out = fftshift(im_out, axis=2)
            im_out = fftshift(im_out, axis=1)

        with tf.device('/gpu:0'):
            im_out = tf.fft2d(im_out) / fftscale

        if len(im.get_shape()) == 5:
            im_out = fftshift(im_out, axis=4)
            im_out = fftshift(im_out, axis=3)
            im_out = tf.transpose(im_out, [0, 3, 4, 1, 2])
        elif len(im.get_shape()) == 4:
            im_out = fftshift(im_out, axis=3)
            im_out = fftshift(im_out, axis=2)
            im_out = tf.transpose(im_out, [0, 2, 3, 1])
        else:
            im_out = fftshift(im_out, axis=2)
            im_out = fftshift(im_out, axis=1)
            im_out = tf.transpose(im_out, [1, 2, 0])

    return im_out


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


def sensemap_model(x, sensemap, name='sensemap_model', do_transpose=False):
    """Apply sensitivity maps."""
    with tf.variable_scope(name):
        if do_transpose:
            x_shape = x.get_shape().as_list()
            x = tf.expand_dims(x, axis=-2)
            x = tf.multiply(tf.conj(sensemap), x)
            x = tf.reduce_sum(x, axis=-1)
        else:
            x = tf.expand_dims(x, axis=-1)
            x = tf.multiply(x, sensemap)
            x = tf.reduce_sum(x, axis=-2)
    return x


def model_forward(x, sensemap, name='model_forward'):
    """Apply forward model.

    Image domain to k-space domain.
    """
    with tf.variable_scope(name):
        if sensemap is not None:
            x = sensemap_model(x, sensemap, do_transpose=False)
        x = fft2c(x)
    return x


def model_transpose(x, sensemap, name='model_transpose'):
    """Apply transpose model.

    k-Space domain to image domain
    """
    with tf.variable_scope(name):
        x = ifft2c(x)
        if sensemap is not None:
            x = sensemap_model(x, sensemap, do_transpose=True)
    return x
