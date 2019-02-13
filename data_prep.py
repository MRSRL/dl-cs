"""Data preparation for training."""
import tensorflow as tf
import os
import logging
import mridata
import ismrmrd
from tqdm import tqdm
import numpy as np
import subprocess
import argparse

from utils import tfmri
from utils import fftc
from utils import cfl

BIN_BART = "bart"


def download_mridata_org_dataset(filename_txt, dir_output):
    """Download datasets from mridata.org if needed"""
    if os.path.isdir(dir_output):
        logging.warning(
            "Downloading data mridata.org to existing directory {}...".format(dir_output))
    else:
        os.makedirs(dir_output)
        logging.info(
            "Downloading data from mridata.org to {}...".format(dir_output))

    uuids = open(filename_txt).read().splitlines()
    for uuid in uuids:
        if not os.path.exists("{}/{}.h5".format(dir_output, uuid)):
            mridata.download(uuid, folder=dir_output)


def ismrmrd_to_np(filename, verbose=False):
    """Read ISMRMRD data file to numpy array"""
    logging.debug("Loading file {}...".format(filename))
    dataset = ismrmrd.Dataset(filename, create_if_needed=False)
    header = ismrmrd.xsd.CreateFromDocument(dataset.read_xml_header())
    num_kx = header.encoding[0].encodedSpace.matrixSize.x
    num_ky = header.encoding[0].encodingLimits.kspace_encoding_step_1.maximum
    num_slices = header.encoding[0].encodingLimits.slice.maximum + 1
    num_channels = header.acquisitionSystemInformation.receiverChannels

    try:
        rec_std = dataset.read_array('rec_std', 0)
        rec_weight = 1.0 / (rec_std ** 2)
        rec_weight = np.sqrt(rec_weight / np.sum(rec_weight))
        logging.debug("  Using rec std...")
    except Exception:
        rec_weight = np.ones(num_channels)
    opt_mat = np.diag(rec_weight)
    kspace = np.zeros([num_channels, num_slices, num_ky,
                       num_kx], dtype=np.complex64)
    num_acq = dataset.number_of_acquisitions()

    def wrap(x): return x
    if logging.getLogger().getEffectiveLevel() is logging.DEBUG:
        wrap = tqdm
    for i in wrap(range(num_acq)):
        acq = dataset.read_acquisition(i)
        i_ky = acq.idx.kspace_encode_step_1  # pylint: disable=E1101
        # i_kz = acq.idx.kspace_encode_step_2 # pylint: disable=E1101
        i_slice = acq.idx.slice             # pylint: disable=E1101
        data = np.matmul(opt_mat.T, acq.data)
        kspace[:, i_slice, i_ky, :] = data

    dataset.close()

    return kspace


def ismrmrd_to_cfl(dir_input, dir_output):
    """Convert ISMRMRD files to CFL files"""
    if os.path.isdir(dir_output):
        logging.warning(
            "Writing cfl data to existing directory {}...".format(dir_output))
    else:
        os.makedirs(dir_output)
        logging.info(
            "Writing cfl data to {}...".format(dir_output))

    filelist = os.listdir(dir_input)

    logging.info("Converting files from ISMRMD to CFL...")
    for filename in filelist:
        # logging.info("  {}...".format(filename))
        file_input = os.path.join(dir_input, filename)
        file_output = os.path.join(dir_output, filename)
        if not os.path.exists(file_output + ".cfl"):
            kspace = ismrmrd_to_np(file_input)
            cfl.write(file_output, kspace)


def create_masks(dir_out, shape_y=320, shape_z=256, verbose=False,
                 acc_y=(1, 2, 3), acc_z=(1, 2, 3),
                 shape_calib=1, variable_density=False, num_repeat=4):
    """Create sampling masks using BART."""
    flags = ""
    file_fmt = "mask_%0.1fx%0.1f_c%d_%02d"
    if variable_density:
        flags = flags + " -v "
        file_fmt = file_fmt + "_vd"

    if not os.path.exists(dir_out):
        os.mkdir(dir_out)

    for a_y in acc_y:
        for a_z in acc_z:
            num_repeat_i = num_repeat
            if (a_y == acc_y[-1]) and (a_z == acc_z[-1]):
                num_repeat_i = num_repeat_i * 2
            for i in range(num_repeat_i):
                if a_y * a_z != 1:
                    random_seed = 1e6 * np.random.random()
                    file_name = file_fmt % (a_y, a_z, shape_calib, i)
                    if verbose:
                        print("creating mask (%s)..." % file_name)
                    file_name = os.path.join(dir_out, file_name)
                    cmd = "%s poisson -C %d -Y %d -Z %d -y %d -z %d -s %d %s %s" % \
                        (BIN_BART, shape_calib, shape_y, shape_z,
                         a_y, a_z, random_seed, flags, file_name)
                    subprocess.check_output(['bash', '-c', cmd])


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def setup_data_tfrecords(dir_in_root, dir_out,
                         data_divide=(.75, .05, .2),
                         min_shape=[80, 180],
                         verbose=False):
    """Setups training data as tfrecords."""
    logging.info("Directory names:")
    logging.info("  Input root:  {}".format(dir_in_root))
    logging.info("  Output root: {}".format(dir_out))

    file_kspace = "kspace"
    file_sensemap = "sensemap"

    case_list = os.listdir(dir_in_root)
    np.random.shuffle(case_list)
    num_cases = len(case_list)

    i_train_1 = np.round(data_divide[0]*num_cases).astype(int)
    i_validate_0 = i_train_1 + 1
    i_validate_1 = np.round(data_divide[1]*num_cases).astype(int) \
        + i_validate_0

    if not os.path.exists(dir_out):
        os.mkdir(dir_out)
    if not os.path.exists(os.path.join(dir_out, "train")):
        os.mkdir(os.path.join(dir_out, "train"))
    if not os.path.exists(os.path.join(dir_out, "validate")):
        os.mkdir(os.path.join(dir_out, "validate"))
    if not os.path.exists(os.path.join(dir_out, "test")):
        os.mkdir(os.path.join(dir_out, "test"))

    i_case = 0
    for case_name in case_list:
        file_kspace_i = os.path.join(dir_in_root, case_name, file_kspace)
        file_sensemap_i = os.path.join(dir_in_root, case_name, file_sensemap)

        if i_case < i_train_1:
            dir_out_i = os.path.join(dir_out, "train")
        elif i_case < i_validate_1:
            dir_out_i = os.path.join(dir_out, "validate")
        else:
            dir_out_i = os.path.join(dir_out, "test")

        if verbose:
            print("Processing [%d] %s..." % (i_case, case_name))
        i_case = i_case + 1

        kspace = np.squeeze(cfl.read(file_kspace_i))
        if (min_shape is None) or (min_shape[0] <= kspace.shape[1] and
                                   min_shape[1] <= kspace.shape[2]):
            if verbose:
                print("  Slice shape: (%d, %d)" %
                      (kspace.shape[1], kspace.shape[2]))
                print("  Num channels: %d" % kspace.shape[0])
            shape_x = kspace.shape[-1]
            kspace = fftc.ifftc(kspace, axis=-1)
            kspace = kspace.astype(np.complex64)

            # if shape_c_out < shape_c:
            #     if verbose:
            #         print("  applying coil compression (%d -> %d)..." %
            #               (shape_c, shape_c_out))
            #     shape_cal = 24
            #     ks_cal = recon.crop(ks, [-1, shape_cal, shape_cal, -1])
            #     ks_cal = np.reshape(ks_cal, [shape_c,
            #                                  shape_cal*shape_cal,
            #                                  shape_x])
            #     cc_mat = coilcomp.calc_gcc_weights_c(ks_cal, shape_c_out)
            #     ks_cc = np.reshape(ks, [shape_c, -1, shape_x])
            #     ks_cc = coilcomp.apply_gcc_weights_c(ks_cc, cc_mat)
            #     ks = np.reshape(ks_cc, [shape_c_out, shape_z, shape_y, shape_x])

            cmd = "%s ecalib -c 1e-9 -m 1 %s %s" % (
                BIN_BART, file_kspace_i, file_sensemap_i)
            if verbose:
                print("  Estimating sensitivity maps (bart espirit)...")
                print("    %s" % cmd)
            subprocess.check_call(["bash", "-c", cmd])
            sensemap = np.squeeze(cfl.read(file_sensemap_i))
            sensemap = np.expand_dims(sensemap, axis=0)
            sensemap = sensemap.astype(np.complex64)

            if verbose:
                print("  Creating tfrecords (%d)..." % shape_x)
            for i_x in range(shape_x):
                file_out = os.path.join(
                    dir_out_i, "%s_x%03d.tfrecords" % (case_name, i_x))
                kspace_x = kspace[:, :, :, i_x]
                sensemap_x = sensemap[:, :, :, :, i_x]

                example = tf.train.Example(features=tf.train.Features(feature={
                    'name': _bytes_feature(str.encode(case_name)),
                    'xslice': _int64_feature(i_x),
                    'ks_shape_x': _int64_feature(kspace.shape[3]),
                    'ks_shape_y': _int64_feature(kspace.shape[2]),
                    'ks_shape_z': _int64_feature(kspace.shape[1]),
                    'ks_shape_c': _int64_feature(kspace.shape[0]),
                    'map_shape_x': _int64_feature(sensemap.shape[4]),
                    'map_shape_y': _int64_feature(sensemap.shape[3]),
                    'map_shape_z': _int64_feature(sensemap.shape[2]),
                    'map_shape_c': _int64_feature(sensemap.shape[1]),
                    'map_shape_m': _int64_feature(sensemap.shape[0]),
                    'ks': _bytes_feature(kspace_x.tostring()),
                    'map': _bytes_feature(sensemap_x.tostring())
                }))

                tf_writer = tf.python_io.TFRecordWriter(file_out)
                tf_writer.write(example.SerializeToString())
                tf_writer.close()


def process_tfrecord(example, num_channels=None, num_maps=None):
    """Process TFRecord to actual tensors."""
    features = tf.parse_single_example(
        example,
        features={
            'name': tf.FixedLenFeature([], tf.string),
            'xslice': tf.FixedLenFeature([], tf.int64),
            'ks_shape_x': tf.FixedLenFeature([], tf.int64),
            'ks_shape_y': tf.FixedLenFeature([], tf.int64),
            'ks_shape_z': tf.FixedLenFeature([], tf.int64),
            'ks_shape_c': tf.FixedLenFeature([], tf.int64),
            'map_shape_x': tf.FixedLenFeature([], tf.int64),
            'map_shape_y': tf.FixedLenFeature([], tf.int64),
            'map_shape_z': tf.FixedLenFeature([], tf.int64),
            'map_shape_c': tf.FixedLenFeature([], tf.int64),
            'map_shape_m': tf.FixedLenFeature([], tf.int64),
            'ks': tf.FixedLenFeature([], tf.string),
            'map': tf.FixedLenFeature([], tf.string)
        }
    )

    name = features['name']
    xslice = tf.cast(features['xslice'], dtype=tf.int32)
    # shape_x = tf.cast(features['shape_x'], dtype=tf.int32)
    ks_shape_y = tf.cast(features['ks_shape_y'], dtype=tf.int32)
    ks_shape_z = tf.cast(features['ks_shape_z'], dtype=tf.int32)
    if num_channels is None:
        ks_shape_c = tf.cast(features['ks_shape_c'], dtype=tf.int32)
    else:
        ks_shape_c = num_channels
    map_shape_y = tf.cast(features['map_shape_y'], dtype=tf.int32)
    map_shape_z = tf.cast(features['map_shape_z'], dtype=tf.int32)
    if num_channels is None:
        map_shape_c = tf.cast(features['map_shape_c'], dtype=tf.int32)
    else:
        map_shape_c = num_channels
    if num_maps is None:
        map_shape_m = tf.cast(features['map_shape_m'], dtype=tf.int32)
    else:
        map_shape_m = num_maps

    with tf.name_scope("kspace"):
        ks_record_bytes = tf.decode_raw(features['ks'], tf.float32)
        image_shape = [ks_shape_c, ks_shape_z, ks_shape_y]
        ks_x = tf.reshape(ks_record_bytes, image_shape + [2])
        ks_x = tfmri.channels_to_complex(ks_x)
        ks_x = tf.reshape(ks_x, image_shape)

    with tf.name_scope("sensemap"):
        map_record_bytes = tf.decode_raw(features['map'], tf.float32)
        map_shape = [map_shape_m * map_shape_c, map_shape_z, map_shape_y]
        map_x = tf.reshape(map_record_bytes, map_shape + [2])
        map_x = tfmri.channels_to_complex(map_x)
        map_x = tf.reshape(map_x, map_shape)

    return name, xslice, ks_x, map_x, map_shape_c


def read_tfrecord_with_sess(tf_sess, filename_tfrecord):
    """Read TFRecord for debugging."""
    tf_reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([filename_tfrecord])
    _, serialized_example = tf_reader.read(filename_queue)
    name, xslice, ks_x, map_x = process_tfrecord(serialized_example)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=tf_sess, coord=coord)
    name, xslice, ks_x, map_x = tf_sess.run([name, xslice, ks_x, map_x])
    coord.request_stop()
    coord.join(threads)

    return {"name": name, "xslice": xslice, "ks": ks_x, "sensemap": map_x}


def read_tfrecord(filename_tfrecord):
    """Read TFRecord for debugging."""
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True  # pylint: disable=E1101
    tf_sess = tf.Session(config=session_config)
    data = read_tfrecord_with_sess(tf_sess, filename_tfrecord)
    tf_sess.close()
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data preparation")
    parser.add_argument("mridata_txt", action="store",
                        help="Text file with mridata.org UUID datasets")
    parser.add_argument("-o", "--output", default="data",
                        help="Output root directory (default: data)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="verbose printing (default: False)")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    dir_mridata_org = os.path.join(args.output, "raw/ismrmrd")
    #download_mridata_org_dataset(args.mridata_txt, dir_mridata_org)

    dir_cfl = os.path.join(args.output, "raw/cfl")
    ismrmrd_to_cfl(dir_mridata_org, dir_cfl)
