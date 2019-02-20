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
import json
import common
import logging
from utils import tfmri

# Data dimensions
tf.app.flags.DEFINE_integer('shape_y', 320, 'Image shape in Y')
tf.app.flags.DEFINE_integer('shape_z', 256, 'Image shape in Z')
tf.app.flags.DEFINE_integer('shape_calib', 10, 'Shape of calibration region')
tf.app.flags.DEFINE_integer(
    'num_channels', 8, 'Number of channels for input datasets.')
tf.app.flags.DEFINE_integer(
    'num_maps', 1, 'Number of eigen maps for input sensitivity maps.')

# For logging
tf.app.flags.DEFINE_string(
    'model_dir', 'summary/model', 'Directory for checkpoints and event logs.')
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
tf.app.flags.DEFINE_integer(
    'unrolled_num_features', 64, 'Number of feature maps in each ResBlock')
tf.app.flags.DEFINE_integer(
    'unrolled_num_resblocks', 3, 'Number of ResBlocks per iteration')
tf.app.flags.DEFINE_boolean(
    'unrolled_share', False, 'Share weights between iterations')
tf.app.flags.DEFINE_boolean(
    'hard_projection', False, 'Turn on/off hard data projection at the end')

# Optimization Flags
tf.app.flags.DEFINE_string('device', '0', 'GPU device to use.')
tf.app.flags.DEFINE_integer(
    'batch_size', 4, 'The number of samples in each batch.')
tf.app.flags.DEFINE_float('loss_l1', 1, 'L1 loss')
tf.app.flags.DEFINE_float('loss_l2', 0, 'L2 loss')
tf.app.flags.DEFINE_float('loss_adv', 0, 'Adversarial loss')
tf.app.flags.DEFINE_integer(
    'adv_steps', 5, 'Steps to train adversarial loss for each recon train step')
tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9, 'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999, 'The exponential decay rate for the 2nd moment estimates.')
tf.app.flags.DEFINE_float(
    'opt_epsilon', 1e-8, 'Epsilon term for the optimizer.')
tf.app.flags.DEFINE_float(
    'learning_rate', 0.01, 'Initial learning rate.')
tf.app.flags.DEFINE_integer(
    'max_steps', None, 'The maximum number of training steps.')

# Dataset Flags
tf.app.flags.DEFINE_string(
    'dir_validate', 'data/tfrecord/validate', 'Directory for validation data (None turns off validation)')
tf.app.flags.DEFINE_string(
    'dir_masks', 'data/masks', 'Directory where masks are located.')
tf.app.flags.DEFINE_string(
    'dir_train', 'data/tfrecord/train', 'Directory where training data are located.')

FLAGS = tf.app.flags.FLAGS

logger = logging.getLogger('recon_train')
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT, None))
logger.addHandler(handler)

class RunTrainOpHooks(tf.train.SessionRunHook):
    """Based on tf.contrib.gan training."""
    def __init__(self, train_op, train_steps):
        self.train_op = train_op
        self.train_steps = train_steps

    def before_run(self, run_context):
        for _ in range(self.train_steps):
            run_context.session.run(self.train_op)


def model_fn(features, labels, mode, params):
    """Main model function to setup training/testing."""
    training = (mode == tf.estimator.ModeKeys.TRAIN)

    adv_scope = 'Adversarial'
    recon_scope = 'ReconNetwork'

    ks_example = features['ks_input']
    mask_example = tfmri.kspace_mask(ks_example, dtype=tf.complex64)
    sensemap = features['sensemap']
    if training:
        mask_recon = features['mask_recon']
    else:
        mask_recon = 1
    image_example = tfmri.model_transpose(ks_example, sensemap)

    image_out, kspace_out, iter_out = model.unrolled_prox(
        ks_example, sensemap,
        num_grad_steps=params['unrolled_steps'],
        resblock_num_features=params['unrolled_num_features'],
        resblock_num_blocks=params['unrolled_num_resblocks'],
        resblock_share=params['unrolled_share'],
        training=training,
        hard_projection=params['hard_projection'],
        mask_output=mask_recon,
        mask=mask_example,
        scope=recon_scope)
    predictions = {'results': image_out}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    ks_truth = labels
    image_truth = tfmri.model_transpose(ks_truth * mask_recon, sensemap)

    with tf.name_scope('loss'):
        loss_total = 0
        loss_l1 = tf.reduce_mean(tf.abs(image_out - image_truth), name='loss-l1')
        loss_l2 = tf.reduce_mean(tf.square(tf.abs(image_out - image_truth)), name='loss-l2')
        if params['loss_l1'] > 0:
            logger.info('Loss: adding l1 loss {}...'.format(params['loss_l1']))
            loss_total += params['loss_l1'] * loss_l1
        if params['loss_l2'] > 0:
            logger.info('Loss: adding l2 loss {}...'.format(params['loss_l2']))
            loss_total += params['loss_l2'] * loss_l2
        tf.summary.scalar('l1', loss_l1)
        tf.summary.scalar('l2', loss_l2)

        if params['loss_adv'] > 0:
            logger.info('Loss: adding adversarial loss {}...'.format(params['loss_adv']))
            adv_truth = model.adversarial(image_truth, training=training, scope=adv_scope)
            adv_recon = model.adversarial(image_out, training=training, scope=adv_scope)
            adv_mse = tf.reduce_mean(tf.square(tf.abs(adv_truth - adv_recon)))
            loss_adv_d = -adv_mse # train as "discriminator"
            loss_adv_g = adv_mse # train as "generator"
            loss_total += params['loss_adv'] * loss_adv_g
            tf.summary.scalar('adv/l2', adv_mse)

    metric_mse = tf.metrics.mean_squared_error(image_truth, image_out)
    metrics = {'mse': metric_mse}

    num_summary_image = params.get('num_summary_image', 0)
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

    image_summary = {'input': image_example,
                     'output': image_out,
                     'truth': image_truth}
    kspace_summary = {'input': features['ks_input'],
                      'output': kspace_out,
                      'truth': ks_truth}

    with tf.name_scope('max'):
        for key in kspace_summary.keys():
            tf.summary.scalar('kspace/' + key, tf.reduce_max(tf.abs(kspace_summary[key])))
        for key in image_summary.keys():
            tf.summary.scalar(key, tf.reduce_max(tf.abs(image_summary[key])))
        tf.summary.scalar('sensemap', tf.reduce_max(tf.abs(sensemap)))

    with tf.name_scope('kspace'):
        summary_kspace = None
        for key in sorted(kspace_summary.keys()):
            summary_tmp = tfmri.sumofsq(kspace_summary[key], keep_dims=True)
            if summary_kspace is None:
                summary_kspace = summary_tmp
            else:
                summary_kspace = tf.concat((summary_kspace, summary_tmp), axis=2)
        summary_kspace = tf.log(summary_kspace + 1e-6)
        tf.summary.image(
            '-'.join(sorted(kspace_summary.keys())),
            summary_kspace, max_outputs=num_summary_image)

    with tf.name_scope('image'):
        summary_image = None
        for key in sorted(image_summary.keys()):
            summary_tmp = tfmri.sumofsq(image_summary[key], keep_dims=True)
            if summary_image is None:
                summary_image = summary_tmp
            else:
                summary_image = tf.concat((summary_image, summary_tmp), axis=2)
        tf.summary.image(
            '-'.join(sorted(image_summary.keys())),
            summary_image, max_outputs=num_summary_image)

    with tf.name_scope('recon'):
        summary_iter = None
        for i in range(params['unrolled_steps']):
            iter_name = 'iter_%02d' % i
            tmp = tfmri.sumofsq(iter_out[iter_name], keep_dims=True)
            if summary_iter is None:
                summary_iter = tmp
            else:
                summary_iter = tf.concat((summary_iter, tmp), axis=2)
            tf.summary.scalar('max/' + iter_name, tf.reduce_max(tmp))
        if summary_iter is not None:
            tf.summary.image(
                'iter/image', summary_iter, max_outputs=params['num_summary_image'])

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_hook = tf.train.SummarySaverHook(
            save_steps=1,
            output_dir=params['dir_validate_results'],
            summary_op=tf.summary.merge_all())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss_total, predictions=predictions,
            evaluation_hooks=[eval_hook], eval_metric_ops=metrics)

    train_op = tf.no_op()
    training_hooks = []

    update_recon_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=recon_scope)
    var_recon = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=recon_scope)
    opt_recon = tf.train.AdamOptimizer(
        params['learning_rate'],
        beta1=params['adam_beta1'],
        beta2=params['adam_beta2'],
        epsilon=params['adam_epsilon'])
    with tf.control_dependencies(update_recon_ops):
        train_recon_op = opt_recon.minimize(
            loss=loss_total, global_step=tf.train.get_global_step(), var_list=var_recon)
    recon_hook = RunTrainOpHooks(train_recon_op, 1)
    training_hooks.insert(0, recon_hook)

    if params['loss_adv'] > 0:
        update_adv_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=adv_scope)
        var_adv = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=adv_scope)

        opt_adv = tf.train.AdamOptimizer(
            params['learning_rate'],
            beta1=params['adam_beta1'],
            beta2=params['adam_beta2'],
            epsilon=params['adam_epsilon'])

        with tf.control_dependencies(update_adv_ops):
            train_adv_op = opt_adv.minimize(
                loss=loss_adv_d, global_step=tf.train.get_global_step(), var_list=var_adv)
        logger.info('Training Adversarial loss: {} for every 1 step'.format(params['adv_steps']))
        adv_hook = RunTrainOpHooks(train_adv_op, params['adv_steps'])
        training_hooks.insert(0, adv_hook)

    logger.info('Number variables:')
    num_var_recon = np.sum([np.prod(v.get_shape().as_list()) for v in var_recon])
    logger.info('  {}: {}'.format(recon_scope, num_var_recon))
    if params['loss_adv'] > 0:
        num_var_adv = np.sum([np.prod(v.get_shape().as_list()) for v in var_adv])
        logger.info('  {}: {}'.format(adv_scope, num_var_adv))

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss_total,
        train_op=train_op,
        training_hooks=training_hooks,
        eval_metric_ops=metrics)


def main(_):
    """Execute main function."""
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.device

    if FLAGS.random_seed >= 0:
        random.seed(FLAGS.random_seed)
        np.random.seed(FLAGS.random_seed)

    tf.logging.set_verbosity(tf.logging.INFO)
    logger.setLevel(logging.INFO)

    out_shape = [FLAGS.shape_z, FLAGS.shape_y]
    dataset_train = data.create_dataset(
        FLAGS.dir_train,
        FLAGS.dir_masks,
        shape_calib=FLAGS.shape_calib,
        num_channels=FLAGS.num_channels,
        num_maps=FLAGS.num_maps,
        batch_size=FLAGS.batch_size,
        out_shape=out_shape)

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True # pylint: disable=E1101
    session_config.allow_soft_placement = True

    dir_val_results = os.path.join(FLAGS.model_dir, 'validate')

    config = tf.estimator.RunConfig(
        log_step_count_steps=FLAGS.log_step_count_steps,
        save_summary_steps=FLAGS.save_summary_steps,
        save_checkpoints_secs=FLAGS.save_checkpoints_secs,
        model_dir=FLAGS.model_dir,
        tf_random_seed=FLAGS.random_seed,
        session_config=session_config)

    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)

    model_params = {'learning_rate': FLAGS.learning_rate,
                    'adam_beta1': FLAGS.adam_beta1,
                    'adam_beta2': FLAGS.adam_beta2,
                    'adam_epsilon': FLAGS.opt_epsilon,
                    'loss_l1': FLAGS.loss_l1,
                    'loss_l2': FLAGS.loss_l2,
                    'loss_adv': FLAGS.loss_adv,
                    'adv_steps': FLAGS.adv_steps,
                    'unrolled_steps': FLAGS.unrolled_steps,
                    'unrolled_num_features': FLAGS.unrolled_num_features,
                    'unrolled_num_resblocks': FLAGS.unrolled_num_resblocks,
                    'unrolled_share': FLAGS.unrolled_share,
                    'hard_projection': FLAGS.hard_projection,
                    'num_summary_image': FLAGS.num_summary_image,
                    'dir_validate_results': dir_val_results}
    with open(os.path.join(FLAGS.model_dir, common.FILENAME_PARAMS), 'w') as fp:
        json.dump(model_params, fp)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn, params=model_params, config=config)

    def _prep_data(dataset):
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels
    train_input_fn = lambda: _prep_data(dataset_train)

    if FLAGS.dir_validate:
        dataset_validate = data.create_dataset(
            FLAGS.dir_validate,
            FLAGS.dir_masks,
            num_channels=FLAGS.num_channels,
            num_maps=FLAGS.num_maps,
            batch_size=FLAGS.batch_size,
            out_shape=out_shape)
        validate_input_fn = lambda: _prep_data(dataset_validate)
        train_spec = tf.estimator.TrainSpec(
            input_fn=train_input_fn, max_steps=FLAGS.max_steps)
        eval_spec = tf.estimator.EvalSpec(
            input_fn=validate_input_fn, steps=1,
            start_delay_secs=10*60, throttle_secs=10*60)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    else:
        estimator.train(input_fn=train_input_fn, max_steps=FLAGS.max_steps)


if __name__ == '__main__':
    tf.app.run()
