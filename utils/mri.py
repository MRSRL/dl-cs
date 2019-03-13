"""Basic MRI reconstruction functions."""
import os
import subprocess
import shutil
import numpy as np
import sigpy.mri
from utils import cfl

BIN_BART = 'bart'


def remove_bart_files(filenames):
    """Remove bart files in list.

    Args:
        filenames: List of bart file names.
    """
    for f in filenames:
        os.remove(f + '.hdr')
        os.remove(f + '.cfl')


def estimate_sense_maps(kspace, calib=20):
    """Estimate sensitivity maps

    ESPIRiT is used if bart exists. Otherwise, use JSENSE in sigpy.

    Args:
        kspace: k-Space data input as [coils, spatial dimensions].
        calib: Calibration region shape in all spatial dimensions.
    Returns:
        Sensitivity maps estimated.
    """
    if shutil.which(BIN_BART):
        flags = '-c 1e-9 -m 1 -r %d' % calib
        randnum = np.random.randint(1e8)
        fileinput = "tmp.in.{}".format(randnum)
        fileoutput = "tmp.out.{}".format(randnum)
        cfl.write(fileinput, kspace)
        cmd = "{} ecalib {} {} {}".format(BIN_BART, flags, fileinput,
                                          fileoutput)
        subprocess.check_output(['bash', '-c', cmd])
        sensemap = np.squeeze(cfl.read(fileoutput))
        remove_bart_files([fileoutput, fileinput])
    else:
        JsenseApp = sigpy.mri.app.JsenseRecon(kspace, ksp_calib_width=calib)
        sensemap = JsenseApp.run()
        del JsenseApp
        sensemap = sensemap.astype(np.complex64)
    return sensemap


def sumofsq(im, axis=0):
    """Compute square root of sum of squares.

    Args:
        im: Raw image.
        axis: Channel axis.
    Returns:
        Square root of sum of squares of input image.
    """
    out = np.sqrt(np.sum(im.real * im.real + im.imag * im.imag, axis=axis))
    return out


def crop_in_dim(im, shape, dim):
    """Centered crop of image in one dimension.

    Args:
        im: Input image.
        shape: Shape to crop to.
        dim: Dimension to perform crop.
    Returns:
        Copy of the input image cropped.
    """
    if shape == im.shape[dim]:
        return im
    if shape > im.shape[dim]:
        return im

    im_shape = im.shape
    tmp_shape = [
        int(np.prod(im_shape[:dim])), im_shape[dim],
        int(np.prod(im_shape[(dim + 1):]))
    ]
    im_out = np.reshape(im, tmp_shape)
    ind0 = (im_shape[dim] - shape) // 2
    ind1 = ind0 + shape
    im_out = im_out[:, ind0:ind1, :].copy()
    im_out = np.reshape(im_out,
                        im_shape[:dim] + (shape, ) + im_shape[(dim + 1):])
    return im_out


def crop(im, out_shape, verbose=False):
    """Centered crop.

    Args:
        im: Image input
        out_shape: Shape to crop input
    Raises:
        TypeError: If number of dimensions of im is not the same as the length of out_shape.
    Returns:
        Cropped copy of the input image.
    """
    if im.ndim != np.size(out_shape):
        raise TypeError(
            'Num dim of input not same as desired shape ({} != {})'.format(
                im.ndim, np.size(out_shape)))
    im_out = im
    for i in range(np.size(out_shape)):
        if out_shape[i] > 0:
            if verbose:
                print(
                    'Crop [%d]: %d to %d' % (i, im_out.shape[i], out_shape[i]))
            im_out = crop_in_dim(im_out, out_shape[i], i)

    return im_out


def zeropad(im, out_shape):
    """Zeropad image.

    Args:
        im: Image input
        out_shape: Shape to crop input
    Raises:
        TypeError: If number of dimensions of im is not the same as the length of out_shape.
    Returns:
        Zero-padded copy of the input image.
    """
    if im.ndim != np.size(out_shape):
        raise TypeError(
            'Num dim of input not same as desired shape ({} != {})'.format(
                im.ndim, np.size(out_shape)))

    pad_shape = []
    for i in range(np.size(out_shape)):
        if out_shape[i] == -1:
            pad_shape_i = [0, 0]
        else:
            pad_start = int((out_shape[i] - im.shape[i]) / 2)
            pad_end = out_shape[i] - im.shape[i] - pad_start
            pad_shape_i = [pad_start, pad_end]

        pad_shape = pad_shape + [pad_shape_i]

    im_out = np.pad(im, pad_shape, 'constant')

    return im_out
