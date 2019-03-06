"""Metrics for testing."""
import numpy as np
import skimage.measure
from utils import mri


def compute_psnr(ref, x):
    """Compute peak to signal to noise ratio."""
    max_val = np.max(np.abs(ref))
    mse = np.mean(np.square(np.abs(x - ref)))
    psnr = 10 * np.log(np.square(max_val) / mse) / np.log(10)
    return psnr


def compute_nrmse(ref, x):
    """Compute normalized root mean square error.

    The norm of reference is used to normalize the metric.
    """
    mse = np.sqrt(np.mean(np.square(np.abs(ref - x))))
    norm = np.sqrt(np.mean(np.square(np.abs(ref))))

    return mse / norm


def compute_ssim(ref, x, data_range=None, sos_axis=None):
    """Compute structural similarity index metric.

    The image is first converted to magnitude image and normalized
    before the metric is computed.
    """
    ref = ref.copy()
    x = x.copy()
    if sos_axis is not None:
        x = mri.sumofsq(x, axis=sos_axis)
        ref = mri.sumofsq(ref, axis=sos_axis)
    x = np.squeeze(x)
    ref = np.squeeze(ref)

    if not data_range:
        data_range = ref.max() - ref.min()

    return skimage.measure.compare_ssim(ref, x,
                                        data_range=data_range,
                                        gaussian_weights=True,
                                        sigma=1.5,
                                        use_sample_covariance=False)
