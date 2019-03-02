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
        diff = np.sum(diff_r ** 2 + diff_i ** 2)
        self.assertTrue(diff < eps)
        self.assertEqual(data_out.shape[-1], 4)

        data_out = tfmri.complex_to_channels(
            data, data_format='channels_first')
        diff_r = data_r - tf.real(data)
        diff_i = data_i - tf.imag(data)
        diff = np.sum(diff_r ** 2 + diff_i ** 2)
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
        diff = np.sum(diff_r ** 2 + diff_i ** 2)
        self.assertTrue(diff < eps)

        data_complex = tfmri.channels_to_complex(
            data, data_format='channels_first')
        diff_r = np.real(data_complex) - data[:, 0:5, ...]
        diff_i = np.imag(data_complex) - data[:, 5:, ...]
        diff = np.sum(diff_r ** 2 + diff_i ** 2)
        self.assertTrue(diff < eps)

        data = tf.random_uniform([10, 10, 2])
        data_complex = tfmri.channels_to_complex(
            data, data_format='channels_first')
        diff_r = np.real(data_complex) - data[0:5, ...]
        diff_i = np.imag(data_complex) - data[5:, ...]
        diff = np.sum(diff_r ** 2 + diff_i ** 2)
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


if __name__ == '__main__':
    unittest.main()
