"""Basic MRI reconstruction functions."""
import numpy as np
import os
import subprocess
import shutil
import sigpy.mri
from utils import cfl

BIN_BART = 'bart'

def remove_bart_files(filenames):
    """Remove bart files in list."""
    for f in filenames:
        os.remove(f + '.hdr')
        os.remove(f + '.cfl')


def estimate_sense_maps(kspace, calib=20):
    """Estimate sensitivity maps

    ESPIRiT is used if bart exists. Otherwise, use JSENSE in sigpy.
    """
    if shutil.which(BIN_BART):
        flags = '-c 1e-9 -m 1 -r %d' % calib
        randnum = np.random.randint(1e8)
        fileinput = "tmp.in.{}".format(randnum)
        fileoutput = "tmp.out.{}".format(randnum)
        cfl.write(fileinput, kspace)
        cmd = "{} ecalib {} {} {}".format(BIN_BART, flags, fileinput, fileoutput)
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

    :param im: raw image
    """
    if axis < 0:
        axis = im.ndim-1
    if axis > im.ndim:
        print('ERROR! Dimension %d invalid for given matrix' % axis)
        return -1

    out = np.sqrt(np.sum(im.real*im.real + im.imag*im.imag, axis=axis))

    return out


def phasecontrast(im, ref, axis=-1, coilaxis=-1):
    """Compute phase contrast."""
    if axis < 0:
        axis = im.ndim-1

    out = np.conj(ref) * im
    if coilaxis >= 0:
        out = np.sum(out, axis=coilaxis)
    out = np.angle(out)

    return out


def fftmod(im, axis=-1):
    """Apply 1 -1 modulation along dimension specified by axis"""
    if axis < 0:
        axis = im.ndim-1

    # generate modulation kernel
    dims = im.shape
    mod = np.ones(np.append(dims[axis], np.ones(len(dims)-1, dtype=int)), dtype=im.dtype)
    mod[1:dims[axis]:2] = -1
    mod = np.transpose(mod, np.append(np.arange(1,len(dims)),0))

    # apply kernel
    tpdims = np.concatenate((np.arange(0,axis), np.arange(axis+1,len(dims)), [axis]))
    out = np.transpose(im, tpdims) # transpose for broadcasting
    out = out * mod
    tpdims = np.concatenate((np.arange(0,axis), [len(dims)-1], np.arange(axis,len(dims)-1)))
    out = np.transpose(out, tpdims) # transpose back to original dims

    return out


def crop_in_dim(im, shape, dim):
    """Centered crop of image."""
    if dim < 0 or dim >= im.ndim:
        print('crop_in_dim> ERROR! Invalid dimension specified!')
        return im
    if shape == im.shape[dim]:
        return im
    if shape > im.shape[dim]:
        # print('crop_in_dim> ERROR! Invalid shape specified!')
        return im

    im_shape = im.shape
    tmp_shape = [int(np.prod(im_shape[:dim])),
                 im_shape[dim],
                 int(np.prod(im_shape[(dim+1):]))]
    im_out = np.reshape(im, tmp_shape)
    ind0 = (im_shape[dim] - shape) // 2
    ind1 = ind0 + shape
    im_out = im_out[:, ind0:ind1, :].copy()
    im_out = np.reshape(im_out, im_shape[:dim] + (shape,) + im_shape[(dim+1):])
    return im_out


def crop(im, out_shape, verbose=False):
    """Centered crop."""
    if im.ndim != np.size(out_shape):
        print('ERROR! Num dim of input image not same as desired shape')
        print('   %d != %d' % (im.ndim, np.size(out_shape)))
        return []

    im_out = im
    for i in range(np.size(out_shape)):
        if out_shape[i] > 0:
            if verbose:
                print('Crop [%d]: %d to %d' % (i, im_out.shape[i],
                                               out_shape[i]))
            im_out = crop_in_dim(im_out, out_shape[i], i)

    return im_out


def zeropad(im, out_shape):
    """Zeropad image."""
    if im.ndim != np.size(out_shape):
        print('ERROR! Num dim of input image not same as desired shape')
        print('   %d != %d' % (im.ndim, np.size(out_shape)))
        return im

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
