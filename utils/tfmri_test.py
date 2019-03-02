import unittest
import numpy as np
import tensorflow as tf
from utils import tfmri

tf.enable_eager_execution()
eps = 1e-7


class TestTFMRI(unittest.TestCase):
    def test_complex_to_channels(self):
        data_r = tf.random_uniform([3, 10, 10, 2])
        data_i = tf.random_uniform([3, 10, 10, 2])
        data = tf.complex(data_r, data_i)
        data_out = tfmri.complex_to_channels(data)
        diff_r = data_r - tf.real(data)
        diff_i = data_i - tf.imag(data)
        diff = np.mean(diff_r ** 2 + diff_i ** 2)
        self.assertTrue(diff < eps)
        self.assertEqual(data_out.shape[-1], 4)

        data_out = tfmri.complex_to_channels(
            data, data_format='channels_first')
        diff_r = data_r - tf.real(data)
        diff_i = data_i - tf.imag(data)
        diff = np.mean(diff_r ** 2 + diff_i ** 2)
        self.assertTrue(diff < eps)
        self.assertEqual(data_out.shape[1], 20)

        with self.assertRaises(TypeError):
            # Input must be complex
            data_out = tfmri.complex_to_channels(data_r)
        with self.assertRaises(TypeError):
            # shape error
            data_r = tf.random_uniform([1, 3, 10, 10, 2])
            data_i = tf.random_uniform([1, 3, 10, 10, 2])
            data = tf.complex(data_r, data_i)
            data_out = tfmri.complex_to_channels(data)
        with self.assertRaises(TypeError):
            # shape error
            data_r = tf.random_uniform([10, 2])
            data_i = tf.random_uniform([10, 2])
            data = tf.complex(data_r, data_i)
            data_out = tfmri.complex_to_channels(data)

    def test_channels_to_complex(self):
        data = tf.random_uniform([2, 10, 10, 2])
        data_complex = tfmri.channels_to_complex(data)
        diff_r = np.real(data_complex) - data[..., 0:1]
        diff_i = np.imag(data_complex) - data[..., 1:]
        diff = np.mean(diff_r ** 2 + diff_i ** 2)
        self.assertTrue(diff < eps)

        data_complex = tfmri.channels_to_complex(
            data, data_format='channels_first')
        diff_r = np.real(data_complex) - data[:, 0:5, ...]
        diff_i = np.imag(data_complex) - data[:, 5:, ...]
        diff = np.mean(diff_r ** 2 + diff_i ** 2)
        self.assertTrue(diff < eps)

        data = tf.random_uniform([10, 10, 2])
        data_complex = tfmri.channels_to_complex(
            data, data_format='channels_first')
        diff_r = np.real(data_complex) - data[0:5, ...]
        diff_i = np.imag(data_complex) - data[5:, ...]
        diff = np.mean(diff_r ** 2 + diff_i ** 2)
        self.assertTrue(diff < eps)

        with self.assertRaises(TypeError):
            # Not enough dimensions
            tfmri.channels_to_complex(tf.random_uniform([10, 10]))
        with self.assertRaises(TypeError):
            # Too many dimensions
            tfmri.channels_to_complex(tf.random_uniform([10, 10, 1, 1, 1]))
        with self.assertRaises(TypeError):
            tfmri.channels_to_complex(tf.random_uniform([10, 10, 1]))
        with self.assertRaises(TypeError):
            tfmri.channels_to_complex(
                tf.random_uniform([5, 10, 1]), data_format='channels_first')
        with self.assertRaises(TypeError):
            tfmri.channels_to_complex(
                tf.random_uniform([1, 5, 10, 1]), data_format='channels_first')

    def test_fftshift(self):
        data = tf.random_uniform([10, 10])
        out = tfmri.fftshift(data, axis=-1)
        out2 = np.fft.fftshift(data, axes=-1)
        diff = np.mean(np.abs(out - out2) ** 2)
        self.assertTrue(diff < eps)

        data = tf.random_uniform([10, 10])
        out = tfmri.fftshift(data, axis=(-2, -1))
        out2 = np.fft.fftshift(data, axes=(-2, -1))
        diff = np.mean(np.abs(out - out2) ** 2)
        self.assertTrue(diff < eps)

    def _fftnc(self, data, axes=(-1,), norm='ortho', transpose=False):
        fdata = np.fft.fftshift(data, axes=axes)
        if transpose:
            fdata = np.fft.ifftn(fdata, axes=axes, norm=norm)
        else:
            fdata = np.fft.fftn(fdata, axes=axes, norm=norm)
        fdata = np.fft.ifftshift(fdata, axes=axes)
        return fdata

    def test_fftc(self):
        shape = [10, 10, 2]
        data = tf.complex(
            tf.random_uniform(shape), tf.random_uniform(shape))
        fdata = tfmri.fftc(data)
        fdata_np = self._fftnc(data, axes=(-2,))
        diff = np.mean(np.abs(fdata_np - fdata) ** 2)
        self.assertTrue(diff < eps)

        fdata = tfmri.fftc(data, data_format='channels_first')
        fdata_np = self._fftnc(data, axes=(-1,))
        diff = np.mean(np.abs(fdata_np - fdata) ** 2)
        self.assertTrue(diff < eps)

        fdata = tfmri.fftc(data, orthonorm=False)
        fdata_np = self._fftnc(data, axes=(-2,), norm=None)
        diff = np.mean(np.abs(fdata_np - fdata) ** 2)
        self.assertTrue(diff < eps)

    def test_ifftc(self):
        shape = [10, 10, 2]
        data = tf.complex(
            tf.random_uniform(shape), tf.random_uniform(shape))
        fdata_np = self._fftnc(data, axes=(-2,), transpose=True)
        fdata = tfmri.ifftc(data)
        diff = np.mean(np.abs(fdata_np - fdata) ** 2)
        self.assertTrue(diff < eps)
        fdata = tfmri.fftc(data, transpose=True)
        diff = np.mean(np.abs(fdata_np - fdata) ** 2)
        self.assertTrue(diff < eps)

        fdata = tfmri.ifftc(data, data_format='channels_first')
        fdata_np = self._fftnc(data, axes=(-1,), transpose=True)
        diff = np.mean(np.abs(fdata_np - fdata) ** 2)
        self.assertTrue(diff < eps)

        fdata = tfmri.ifftc(data, orthonorm=False)
        fdata_np = self._fftnc(data, axes=(-2,), norm=None, transpose=True)
        diff = np.mean(np.abs(fdata_np - fdata) ** 2)
        self.assertTrue(diff < eps)

    def test_fft2c(self):
        shape = [10, 10, 2]
        data = tf.complex(
            tf.random_uniform(shape), tf.random_uniform(shape))
        fdata = tfmri.fft2c(data)
        fdata_np = self._fftnc(data, axes=(-3, -2))
        diff = np.mean(np.abs(fdata_np - fdata) ** 2)
        self.assertTrue(diff < eps)

        fdata = tfmri.fft2c(data, data_format='channels_first')
        fdata_np = self._fftnc(data, axes=(-2, -1))
        diff = np.mean(np.abs(fdata_np - fdata) ** 2)
        self.assertTrue(diff < eps)

        fdata = tfmri.fft2c(data, orthonorm=False)
        fdata_np = self._fftnc(data, axes=(-3, -2), norm=None)
        diff = np.mean(np.abs(fdata_np - fdata) ** 2)
        self.assertTrue(diff < eps)

    def test_ifft2c(self):
        shape = [10, 10, 2]
        data = tf.complex(
            tf.random_uniform(shape), tf.random_uniform(shape))
        fdata_np = self._fftnc(data, axes=(-3, -2), transpose=True)
        fdata = tfmri.ifft2c(data)
        diff = np.mean(np.abs(fdata_np - fdata) ** 2)
        self.assertTrue(diff < eps)
        fdata = tfmri.fft2c(data, transpose=True)
        diff = np.mean(np.abs(fdata_np - fdata) ** 2)
        self.assertTrue(diff < eps)

        fdata = tfmri.ifft2c(data, data_format='channels_first')
        fdata_np = self._fftnc(data, axes=(-2, -1), transpose=True)
        diff = np.mean(np.abs(fdata_np - fdata) ** 2)
        self.assertTrue(diff < eps)

        fdata = tfmri.ifft2c(data, orthonorm=False)
        fdata_np = self._fftnc(data, axes=(-3, -2), norm=None, transpose=True)
        diff = np.mean(np.abs(fdata_np - fdata) ** 2)
        self.assertTrue(diff < eps)

    def test_sumofsq(self):
        shape = [10, 10, 2]
        data = tf.complex(tf.random_uniform(shape), tf.random_uniform(shape))

        sos = tfmri.sumofsq(data)
        sos_np = np.sqrt(np.sum(np.abs(data) ** 2, axis=-1, keepdims=False))
        mse = np.mean(np.abs(sos - sos_np) ** 2)
        self.assertTrue(mse < eps)

        sos = tfmri.sumofsq(data, axis=0, keepdims=True)
        sos_np = np.sqrt(np.sum(np.abs(data) ** 2, axis=0, keepdims=True))
        self.assertEqual(len(sos.shape), len(sos_np.shape))
        mse = np.mean(np.abs(sos - sos_np) ** 2)
        self.assertTrue(mse < eps)


if __name__ == '__main__':
    unittest.main()
