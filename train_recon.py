"""Generic training script that trains a model using a given dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import tensorflow as tf
import model
import data
from utils import tfmri

# Data dimensions
tf.app.flags.DEFINE_integer('shape_y', 320, 'Image shape in Y')
tf.app.flags.DEFINE_integer('shape_z', 256, 'Image shape in Z')
tf.app.flags.DEFINE_integer(
    'num_channels', 8, 'Number of channels for input datasets.')
tf.app.flags.DEFINE_integer(
    'num_maps', 1, 'Number of eigen maps for input sensitivity maps.')

# For logging
tf.app.flags.DEFINE_string(
    'log_root', 'summary', 'Root directory where logs are written to.')
tf.app.flags.DEFINE_string(
    'train_dir', 'train', 'Directory for checkpoints and event logs.')
tf.app.flags.DEFINE_integer(
    'num_summary_image', 4, 'Number of images for summary output')
tf.app.flags.DEFINE_integer(
    'log_step_count_steps', 10, 'The frequency with which logs are print.')
tf.app.flags.DEFINE_integer(
    'save_summary_steps', 100, 'The frequency with which summaries are saved')
tf.app.flags.DEFINE_integer(
    'save_checkpoints_secs', 60, 'The frequency with which the model is saved [s]')
tf.app.flags.DEFINE_integer(
    'random_seed', 1000, 'Seed to initialize random number generators.')

# For model
tf.app.flags.DEFINE_integer(
    'unrolled_steps', 4, 'Number of grad steps for unrolled algorithms')
tf.app.flags.DEFINE_boolean(
    'unrolled_share', False, 'Share weights between iterations')
tf.app.flags.DEFINE_boolean(
    "do_hard_proj", False, "Turn on/off hard data projection at the end")

# Optimization Flags
tf.app.flags.DEFINE_string('device', '0', 'GPU device to use.')
tf.app.flags.DEFINE_integer(
    'batch_size', 4, 'The number of samples in each batch.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9, 'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999, 'The exponential decay rate for the 2nd moment estimates.')
tf.app.flags.DEFINE_float(
    'opt_epsilon', 1e-8, 'Epsilon term for the optimizer.')
tf.app.flags.DEFINE_float(
    'learning_rate', 0.001, 'Initial learning rate.')
tf.app.flags.DEFINE_integer(
    'max_steps', None, 'The maximum number of training steps.')

# Dataset Flags
tf.app.flags.DEFINE_string('mask_path', 'masks',
                           'Directory where masks are located.')
tf.app.flags.DEFINE_string('train_path', 'train',
                           'Sub directory where training data are located.')
tf.app.flags.DEFINE_string('dataset_dir', 'dataset',
                           'The directory where the dataset files are stored.')

FLAGS = tf.app.flags.FLAGS


def model_fn(features, labels, mode, params):
    """Main model function to setup training/testing."""
    ks_truth = labels
    ks_example = features['ks_input']
    mask_example = tfmri.kspace_mask(ks_example, dtype=tf.complex64)
    sensemap = features['sensemap']
    mask_recon = features['mask_recon']
    image_truth = tfmri.model_transpose(ks_truth * mask_recon, sensemap)
    image_example = tfmri.model_transpose(ks_example, sensemap)

    image_out, kspace_out, iter_out = model.unroll_ista(
        ks_example, sensemap,
        num_grad_steps=params["unrolled_steps"],
        resblock_num_features=128,
        resblock_num_blocks=3,
        resblock_share=params["unrolled_share"],
        training=True,
        hard_projection=params["hard_projection"],
        mask_output=mask_recon,
        mask=mask_example)
    predictions = {'results': image_out}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    with tf.name_scope("loss"):
        loss_l1 = tf.reduce_mean(tf.abs(image_out - image_truth), name="loss-l1")
        loss_l2 = tf.reduce_mean(tf.square(tf.abs(image_out - image_truth)), name="loss-l2")
        tf.summary.scalar("l1", loss_l1)
        tf.summary.scalar("l2", loss_l2)
        loss = loss_l1

    metric_mse = tf.metrics.mean_squared_error(image_truth, image_out)
    metrics = {"mse": metric_mse}

    num_summary_image = params.get("num_summary_image", 0)
    with tf.name_scope('mask'):
        summary_mask = tfmri.sumofsq(mask_example, keep_dims=True)
        tf.summary.image('mask', summary_mask, max_outputs=num_summary_image)
    with tf.name_scope('sensemap'):
        summary_truth = tf.transpose(sensemap, [0, 3, 1, 4, 2])
        summary_truth = tf.reshape(
            summary_truth,
            [tf.shape(summary_truth)[0],
             tf.reduce_prod(tf.shape(summary_truth)[1:3]),
             tf.reduce_prod(tf.shape(summary_truth)[3:]), 1])
        tf.summary.image(
            'mag', tf.abs(summary_truth), max_outputs=num_summary_image)
        tf.summary.image(
            'phase', tf.angle(summary_truth), max_outputs=num_summary_image)

    image_summary = {"input": image_example,
                     "output": image_out,
                     "truth": image_truth}
    kspace_summary = {"input": features['ks_input'],
                      "output": kspace_out,
                      "truth": ks_truth}

    with tf.name_scope("max"):
        for key in kspace_summary.keys():
            tf.summary.scalar("kspace/" + key, tf.reduce_max(tf.abs(kspace_summary[key])))
        for key in image_summary.keys():
            tf.summary.scalar(key, tf.reduce_max(tf.abs(image_summary[key])))
        tf.summary.scalar('sensemap', tf.reduce_max(tf.abs(sensemap)))

    with tf.name_scope("kspace"):
        summary_kspace = None
        for key in sorted(kspace_summary.keys()):
            summary_tmp = tfmri.sumofsq(kspace_summary[key], keep_dims=True)
            if summary_kspace is None:
                summary_kspace = summary_tmp
            else:
                summary_kspace = tf.concat((summary_kspace, summary_tmp), axis=2)
        summary_kspace = tf.log(summary_kspace + 1e-6)
        tf.summary.image(
            "-".join(sorted(kspace_summary.keys())),
            summary_kspace, max_outputs=num_summary_image)

    with tf.name_scope("image"):
        summary_image = None
        for key in sorted(image_summary.keys()):
            summary_tmp = tfmri.sumofsq(image_summary[key], keep_dims=True)
            if summary_image is None:
                summary_image = summary_tmp
            else:
                summary_image = tf.concat((summary_image, summary_tmp), axis=2)
        tf.summary.image(
            "-".join(sorted(image_summary.keys())),
            summary_image, max_outputs=num_summary_image)

    with tf.name_scope("recon"):
        summary_iter = None
        for i in range(params["unrolled_steps"]):
            iter_name = "iter_%02d" % i
            tmp = tfmri.sumofsq(iter_out[iter_name], keep_dims=True)
            if summary_iter is None:
                summary_iter = tmp
            else:
                summary_iter = tf.concat((summary_iter, tmp), axis=2)
            tf.summary.scalar('max/' + iter_name, tf.reduce_max(tmp))
        if summary_iter is not None:
            tf.summary.image(
                'iter/image', summary_iter, max_outputs=params['num_summary_image'])

    optimizer = tf.train.AdamOptimizer(
        params["learning_rate"],
        beta1=params["adam_beta1"],
        beta2=params["adam_beta2"],
        epsilon=params["adam_epsilon"])
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)


def main(_):
    """Execute main function."""
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.device

    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with '
                         + '--dataset_dir')

    if FLAGS.random_seed >= 0:
        random.seed(FLAGS.random_seed)
        np.random.seed(FLAGS.random_seed)

    tf.logging.set_verbosity(tf.logging.INFO)

    tf.logging.info("Preparing dataset...")
    out_shape = [FLAGS.shape_z, FLAGS.shape_y]
    train_dataset = data.create_dataset(
        os.path.join(FLAGS.dataset_dir, "train"),
        FLAGS.mask_path,
        num_channels=FLAGS.num_channels,
        num_maps=FLAGS.num_maps,
        batch_size=FLAGS.batch_size,
        out_shape=out_shape)

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True # pylint: disable=E1101
    session_config.allow_soft_placement = True
    config = tf.estimator.RunConfig(
        log_step_count_steps=FLAGS.log_step_count_steps,
        save_summary_steps=FLAGS.save_summary_steps,
        save_checkpoints_secs=FLAGS.save_checkpoints_secs,
        model_dir=os.path.join(FLAGS.log_root, FLAGS.train_dir),
        tf_random_seed=FLAGS.random_seed,
        session_config=session_config)

    model_params = {"learning_rate": FLAGS.learning_rate,
                    "adam_beta1": FLAGS.adam_beta1,
                    "adam_beta2": FLAGS.adam_beta2,
                    "adam_epsilon": FLAGS.opt_epsilon,
                    "unrolled_steps": FLAGS.unrolled_steps,
                    "unrolled_share": FLAGS.unrolled_share,
                    "hard_projection": FLAGS.do_hard_proj,
                    "num_summary_image": FLAGS.num_summary_image}

    estimator = tf.estimator.Estimator(
        model_fn=model_fn, params=model_params, config=config)

    def input_fn():
        """Create input for estimator."""
        train_iterator = train_dataset.make_one_shot_iterator()
        features, labels = train_iterator.get_next()
        return features, labels

    estimator.train(input_fn=input_fn, max_steps=FLAGS.max_steps)


if __name__ == '__main__':
    tf.app.run()
