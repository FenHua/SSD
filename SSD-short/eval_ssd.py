from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf

import numpy as np

from net import ssd_net

from dataset import dataset_common
from preprocessing import ssd_preprocessing
from utility import anchor_manipulator
from utility import scaffolds

# 硬件相关的配置
tf.app.flags.DEFINE_integer(
    'num_readers', 8,
    'The number of parallel readers that read data from the dataset.')  # 读取数据并行数
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 24,
    'The number of threads used to create the batches.')  # 创建batch数据的线程数量
tf.app.flags.DEFINE_integer(
    'num_cpu_threads', 0,
    'The number of cpu cores used to train.')  # CPU用来训练的核数
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 1., 'GPU memory fraction to use.')  # gpu显存使用量
# scaffold related configuration
tf.app.flags.DEFINE_string(
    'data_dir', '/home/yhq/Desktop/SSD-short/dataset/tfrecords',
    'The directory where the dataset input data is stored.')  # 数据集目录
tf.app.flags.DEFINE_integer(
    'num_classes', 21, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string(
    'model_dir', '/home/yhq/Desktop/SSD-short/logs/',
    'The directory where the model will be stored.')
tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are printed.')   # 每隔一定步数输出日志
tf.app.flags.DEFINE_integer(
    'save_summary_steps', 500,
    'The frequency with which summaries are saved, in seconds.')  # 每隔一定时间保存日志
# 模型相关的配置
tf.app.flags.DEFINE_integer(
    'train_image_size', 300,
    'The size of the input image for the model to use.')
tf.app.flags.DEFINE_integer(
    'train_epochs', 1,
    'The number of epochs to use for training.')  # 训练的epoch数
tf.app.flags.DEFINE_integer(
    'batch_size', 1,
    'Batch size for training and evaluation.')  # batch_size大小
tf.app.flags.DEFINE_string(
    'data_format', 'channels_last', # 'channels_first' or 'channels_last'
    'A flag to override the data format used in the model. channels_first '
    'provides a performance boost on GPU but is not always compatible '
    'with CPU. If left unspecified, the data format will be chosen '
    'automatically based on whether TensorFlow was built for CPU or GPU.')  # 数据格式，注意CPU与GPU不同
tf.app.flags.DEFINE_float(
    'negative_ratio', 3., 'Negative ratio in the loss function.')  # loss函数中负样本权重比例
tf.app.flags.DEFINE_float(
    'match_threshold', 0.5, 'Matching threshold in the loss function.')  # IOU阈值
tf.app.flags.DEFINE_float(
    'neg_threshold', 0.5, 'Matching threshold for the negtive examples in the loss function.')  # 负样本的判别阈值
tf.app.flags.DEFINE_float(
    'select_threshold', 0.01, 'Class-specific confidence score threshold for selecting a box.')  # 判断一个box存在目标的阈值
tf.app.flags.DEFINE_float(
    'min_size', 0.03, 'The min size of bboxes to keep.')  # 最小的boxes
tf.app.flags.DEFINE_float(
    'nms_threshold', 0.45, 'Matching threshold in NMS algorithm.')  # NMS阈值
tf.app.flags.DEFINE_integer(
    'nms_topk', 200, 'Number of total object to keep after NMS.')  # NMS后保留top_k
tf.app.flags.DEFINE_integer(
    'keep_topk', 400, 'Number of total object to keep for each image before nms.')  # NMS前保留top_k
# 优化器相关配置
tf.app.flags.DEFINE_float(
    'weight_decay', 5e-4, 'The weight decay on the model weights.')  # 衰减系数
# 检查点文件的相关配置
tf.app.flags.DEFINE_string(
    'checkpoint_path', '/home/yhq/Desktop/SSD-short/model',
    'The path to a checkpoint from which to fine-tune.')  # 用来fine的最初的模型检查点文件
tf.app.flags.DEFINE_string(
    'model_scope', 'ssd300',
    'Model scope name used to replace the name_scope in checkpoint.')  # 模型作用位置的名称
FLAGS = tf.app.flags.FLAGS
#CUDA_VISIBLE_DEVICES


def get_checkpoint():
    if tf.train.latest_checkpoint(FLAGS.model_dir):
        tf.logging.info('Ignoring --checkpoint_path because a checkpoint already exists in %s' % FLAGS.model_dir)
        return None
    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)  # 最新保存的检查点
    else:
        checkpoint_path = FLAGS.checkpoint_path  # 非文件夹时操作
    return checkpoint_path


global_anchor_info = dict()


def input_pipeline(dataset_pattern='train-*', is_training=True, batch_size=FLAGS.batch_size):
    # 数据输入管道
    def input_fn():
        out_shape = [FLAGS.train_image_size] * 2
        anchor_creator = anchor_manipulator.AnchorCreator(out_shape,
                                                    layers_shapes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
                                                    anchor_scales = [(0.1,), (0.2,), (0.375,), (0.55,), (0.725,), (0.9,)],
                                                    extra_anchor_scales = [(0.1414,), (0.2739,), (0.4541,), (0.6315,), (0.8078,), (0.9836,)],
                                                    anchor_ratios = [(1., 2., .5), (1., 2., 3., .5, 0.3333), (1., 2., 3., .5, 0.3333), (1., 2., 3., .5, 0.3333), (1., 2., .5), (1., 2., .5)],
                                                    layer_steps = [8, 16, 32, 64, 100, 300])  # 获取anchor
        all_anchors, all_num_anchors_depth, all_num_anchors_spatial = anchor_creator.get_all_anchors()
        num_anchors_per_layer = []
        for ind in range(len(all_anchors)):
            num_anchors_per_layer.append(all_num_anchors_depth[ind] * all_num_anchors_spatial[ind])  # 每一层的anchor数量
        anchor_encoder_decoder = anchor_manipulator.AnchorEncoder(allowed_borders = [1.0] * 6,
                                                            positive_threshold = FLAGS.match_threshold,
                                                            ignore_threshold = FLAGS.neg_threshold,
                                                            prior_scaling=[0.1, 0.1, 0.2, 0.2])  # anchor编解码器
        # 图片系列处理
        image_preprocessing_fn = lambda image_, labels_, bboxes_ : ssd_preprocessing.preprocess_image(image_, labels_, bboxes_, out_shape, is_training=is_training, data_format=FLAGS.data_format, output_rgb=False)
        # anchor编码
        anchor_encoder_fn = lambda glabels_, gbboxes_: anchor_encoder_decoder.encode_all_anchors(glabels_, gbboxes_, all_anchors, all_num_anchors_depth, all_num_anchors_spatial)
        image, filename, shape, loc_targets, cls_targets, match_scores = dataset_common.slim_get_batch(FLAGS.num_classes,
                                                                                batch_size,
                                                                                ('train' if is_training else 'val'),
                                                                                os.path.join(FLAGS.data_dir, dataset_pattern),
                                                                                FLAGS.num_readers,
                                                                                FLAGS.num_preprocessing_threads,
                                                                                image_preprocessing_fn,
                                                                                anchor_encoder_fn,
                                                                                num_epochs=FLAGS.train_epochs,
                                                                                is_training=is_training)  # 获取batch大小的数据
        global global_anchor_info
        global_anchor_info = {'decode_fn': lambda pred : anchor_encoder_decoder.decode_all_anchors(pred, num_anchors_per_layer),
                            'num_anchors_per_layer': num_anchors_per_layer,
                            'all_num_anchors_depth': all_num_anchors_depth }
        return {'image': image, 'filename': filename, 'shape': shape, 'loc_targets': loc_targets, 'cls_targets': cls_targets, 'match_scores': match_scores}, None
    return input_fn


def modified_smooth_l1(bbox_pred, bbox_targets, bbox_inside_weights=1., bbox_outside_weights=1., sigma=1.):
    """
        ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
        SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                      |x| - 0.5 / sigma^2,    otherwise
        修正loss函数
    """
    with tf.name_scope('smooth_l1', [bbox_pred, bbox_targets]):
        sigma2 = sigma * sigma

        inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))

        smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
        smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
        smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
        smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                                  tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))

        outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)
        return outside_mul


def select_bboxes(scores_pred, bboxes_pred, num_classes, select_threshold):
    # 根据阈值选择出bboxes
    selected_bboxes = {}
    selected_scores = {}
    with tf.name_scope('select_bboxes', [scores_pred, bboxes_pred]):
        for class_ind in range(1, num_classes):
            class_scores = scores_pred[:, class_ind]
            select_mask = class_scores > select_threshold
            select_mask = tf.cast(select_mask, tf.float32)
            selected_bboxes[class_ind] = tf.multiply(bboxes_pred, tf.expand_dims(select_mask, axis=-1))
            selected_scores[class_ind] = tf.multiply(class_scores, select_mask)
    return selected_bboxes, selected_scores


def clip_bboxes(ymin, xmin, ymax, xmax, name):
    # 裁剪操作
    with tf.name_scope(name, 'clip_bboxes', [ymin, xmin, ymax, xmax]):
        ymin = tf.maximum(ymin, 0.)
        xmin = tf.maximum(xmin, 0.)
        ymax = tf.minimum(ymax, 1.)
        xmax = tf.minimum(xmax, 1.)
        ymin = tf.minimum(ymin, ymax)
        xmin = tf.minimum(xmin, xmax)
        return ymin, xmin, ymax, xmax


def filter_bboxes(scores_pred, ymin, xmin, ymax, xmax, min_size, name):
    # 根据bboxes大小进行过滤
    with tf.name_scope(name, 'filter_bboxes', [scores_pred, ymin, xmin, ymax, xmax]):
        width = xmax - xmin
        height = ymax - ymin
        filter_mask = tf.logical_and(width > min_size, height > min_size)
        filter_mask = tf.cast(filter_mask, tf.float32)
        return tf.multiply(ymin, filter_mask), tf.multiply(xmin, filter_mask), \
                tf.multiply(ymax, filter_mask), tf.multiply(xmax, filter_mask), tf.multiply(scores_pred, filter_mask)


def sort_bboxes(scores_pred, ymin, xmin, ymax, xmax, keep_topk, name):
    # 根据score值进行排序，并返回top_k个数据
    with tf.name_scope(name, 'sort_bboxes', [scores_pred, ymin, xmin, ymax, xmax]):
        cur_bboxes = tf.shape(scores_pred)[0]
        scores, idxes = tf.nn.top_k(scores_pred, k=tf.minimum(keep_topk, cur_bboxes), sorted=True)

        ymin, xmin, ymax, xmax = tf.gather(ymin, idxes), tf.gather(xmin, idxes), tf.gather(ymax, idxes), tf.gather(xmax, idxes)

        paddings_scores = tf.expand_dims(tf.stack([0, tf.maximum(keep_topk-cur_bboxes, 0)], axis=0), axis=0)

        return tf.pad(ymin, paddings_scores, "CONSTANT"), tf.pad(xmin, paddings_scores, "CONSTANT"),\
                tf.pad(ymax, paddings_scores, "CONSTANT"), tf.pad(xmax, paddings_scores, "CONSTANT"),\
                tf.pad(scores, paddings_scores, "CONSTANT")


def nms_bboxes(scores_pred, bboxes_pred, nms_topk, nms_threshold, name):
    # 非极大值抑制NMS
    with tf.name_scope(name, 'nms_bboxes', [scores_pred, bboxes_pred]):
        idxes = tf.image.non_max_suppression(bboxes_pred, scores_pred, nms_topk, nms_threshold)
        return tf.gather(scores_pred, idxes), tf.gather(bboxes_pred, idxes)


def parse_by_class(cls_pred, bboxes_pred, num_classes, select_threshold, min_size, keep_topk, nms_topk, nms_threshold):
    # 获取最终选取的一系列bboxes以及相应的置信度
    with tf.name_scope('select_bboxes', [cls_pred, bboxes_pred]):
        scores_pred = tf.nn.softmax(cls_pred)
        selected_bboxes, selected_scores = select_bboxes(scores_pred, bboxes_pred, num_classes, select_threshold)
        for class_ind in range(1, num_classes):
            ymin, xmin, ymax, xmax = tf.unstack(selected_bboxes[class_ind], 4, axis=-1)
            ymin, xmin, ymax, xmax = clip_bboxes(ymin, xmin, ymax, xmax, 'clip_bboxes_{}'.format(class_ind))
            ymin, xmin, ymax, xmax, selected_scores[class_ind] = filter_bboxes(selected_scores[class_ind],
                                                ymin, xmin, ymax, xmax, min_size, 'filter_bboxes_{}'.format(class_ind))
            ymin, xmin, ymax, xmax, selected_scores[class_ind] = sort_bboxes(selected_scores[class_ind],
                                                ymin, xmin, ymax, xmax, keep_topk, 'sort_bboxes_{}'.format(class_ind))
            selected_bboxes[class_ind] = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
            selected_scores[class_ind], selected_bboxes[class_ind] = nms_bboxes(selected_scores[class_ind], selected_bboxes[class_ind], nms_topk, nms_threshold, 'nms_bboxes_{}'.format(class_ind))
        return selected_bboxes, selected_scores


def ssd_model_fn(features, labels, mode, params):
    # SSD 模型的核心部分
    filename = features['filename']  #获取图片名称
    shape = features['shape']  # 预测结果的大小
    loc_targets = features['loc_targets']  # 预测目标的位置
    cls_targets = features['cls_targets']  # 预测结果的分类情况
    match_scores = features['match_scores']  # 预测结果的类别置信度
    features = features['image']  # 特征图
    global global_anchor_info
    decode_fn = global_anchor_info['decode_fn']  # 解码器
    num_anchors_per_layer = global_anchor_info['num_anchors_per_layer']  # 每一层的anchor
    all_num_anchors_depth = global_anchor_info['all_num_anchors_depth']  # 所有anchor的数量
    with tf.variable_scope(params['model_scope'], default_name=None, values=[features], reuse=tf.AUTO_REUSE):
        backbone = ssd_net.VGG16Backbone(params['data_format'])  # 模型恢复
        feature_layers = backbone.forward(features, training=(mode == tf.estimator.ModeKeys.TRAIN))  # 输出的特征层
        location_pred, cls_pred = ssd_net.multibox_head(feature_layers, params['num_classes'], all_num_anchors_depth, data_format=params['data_format'])  # 检测结果
        if params['data_format'] == 'channels_first':
            # GPU 结果进行转换
            cls_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in cls_pred]
            location_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in location_pred]
        cls_pred = [tf.reshape(pred, [tf.shape(features)[0], -1, params['num_classes']]) for pred in cls_pred]  # 预测的类别信息
        location_pred = [tf.reshape(pred, [tf.shape(features)[0], -1, 4]) for pred in location_pred]  # 预测的位置信息
        cls_pred = tf.concat(cls_pred, axis=1)
        location_pred = tf.concat(location_pred, axis=1)
        cls_pred = tf.reshape(cls_pred, [-1, params['num_classes']])
        location_pred = tf.reshape(location_pred, [-1, 4])
    with tf.device('/cpu:0'):
        bboxes_pred = decode_fn(location_pred)   # 解码
        bboxes_pred = tf.concat(bboxes_pred, axis=0)
        selected_bboxes, selected_scores = parse_by_class(cls_pred, bboxes_pred,
                                                        params['num_classes'], params['select_threshold'], params['min_size'],
                                                        params['keep_topk'], params['nms_topk'], params['nms_threshold'])  # 筛选

    predictions = {'filename': filename, 'shape': shape }
    for class_ind in range(1, params['num_classes']):
        predictions['scores_{}'.format(class_ind)] = tf.expand_dims(selected_scores[class_ind], axis=0)  # 增加一维
        predictions['bboxes_{}'.format(class_ind)] = tf.expand_dims(selected_bboxes[class_ind], axis=0)
    flaten_cls_targets = tf.reshape(cls_targets, [-1])
    flaten_match_scores = tf.reshape(match_scores, [-1])
    flaten_loc_targets = tf.reshape(loc_targets, [-1, 4])
    # 每一个正样本有对应的label
    positive_mask = flaten_cls_targets > 0
    n_positives = tf.count_nonzero(positive_mask)
    batch_n_positives = tf.count_nonzero(cls_targets, -1)
    batch_negtive_mask = tf.equal(cls_targets, 0)
    batch_n_negtives = tf.count_nonzero(batch_negtive_mask, -1)
    batch_n_neg_select = tf.cast(params['negative_ratio'] * tf.cast(batch_n_positives, tf.float32), tf.int32)
    batch_n_neg_select = tf.minimum(batch_n_neg_select, tf.cast(batch_n_negtives, tf.int32))
    # 难分样本的选取
    predictions_for_bg = tf.nn.softmax(tf.reshape(cls_pred, [tf.shape(features)[0], -1, params['num_classes']]))[:, :, 0]
    prob_for_negtives = tf.where(batch_negtive_mask, 0. - predictions_for_bg, 0. - tf.ones_like(predictions_for_bg))
    topk_prob_for_bg, _ = tf.nn.top_k(prob_for_negtives, k=tf.shape(prob_for_negtives)[1])
    score_at_k = tf.gather_nd(topk_prob_for_bg, tf.stack([tf.range(tf.shape(features)[0]), batch_n_neg_select - 1], axis=-1))
    selected_neg_mask = prob_for_negtives >= tf.expand_dims(score_at_k, axis=-1)
    # 包括所选择的正负样本
    final_mask = tf.stop_gradient(tf.logical_or(tf.reshape(tf.logical_and(batch_negtive_mask, selected_neg_mask), [-1]), positive_mask))
    total_examples = tf.count_nonzero(final_mask)
    cls_pred = tf.boolean_mask(cls_pred, final_mask)
    location_pred = tf.boolean_mask(location_pred, tf.stop_gradient(positive_mask))
    flaten_cls_targets = tf.boolean_mask(tf.clip_by_value(flaten_cls_targets, 0, params['num_classes']), final_mask)
    flaten_loc_targets = tf.stop_gradient(tf.boolean_mask(flaten_loc_targets, positive_mask))
    # 计算loss，包括softmax交叉熵和L2正则化
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=flaten_cls_targets, logits=cls_pred) * (params['negative_ratio'] + 1.)
    tf.identity(cross_entropy, name='cross_entropy_loss')
    tf.summary.scalar('cross_entropy_loss', cross_entropy)
    loc_loss = modified_smooth_l1(location_pred, flaten_loc_targets, sigma=1.)
    loc_loss = tf.reduce_mean(tf.reduce_sum(loc_loss, axis=-1), name='location_loss')
    tf.summary.scalar('location_loss', loc_loss)
    tf.losses.add_loss(loc_loss)
    # 添加权重衰减到loss函数中
    total_loss = tf.add(cross_entropy, loc_loss, name='total_loss')
    cls_accuracy = tf.metrics.accuracy(flaten_cls_targets, tf.argmax(cls_pred, axis=-1))
    tf.identity(cls_accuracy[1], name='cls_accuracy')
    tf.summary.scalar('cls_accuracy', cls_accuracy[1])
    summary_hook = tf.train.SummarySaverHook(save_steps=params['save_summary_steps'],
                                        output_dir=params['summary_dir'],
                                        summary_op=tf.summary.merge_all())
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
              mode=mode,
              predictions=predictions,
              prediction_hooks=[summary_hook],
              loss=None, train_op=None)
    else:
        raise ValueError('This script only support "PREDICT" mode!')


def parse_comma_list(args):
    return [float(s.strip()) for s in args.split(',')]  # 对输入数据进行分割


def main(_):
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)  # GPU设置
    # 配置信息
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, intra_op_parallelism_threads=FLAGS.num_cpu_threads, inter_op_parallelism_threads=FLAGS.num_cpu_threads, gpu_options=gpu_options)
    # 更新配置文件信息
    run_config = tf.estimator.RunConfig().replace(
                                        save_checkpoints_secs=None).replace(
                                        save_checkpoints_steps=None).replace(
                                        save_summary_steps=FLAGS.save_summary_steps).replace(
                                        keep_checkpoint_max=5).replace(
                                        log_step_count_steps=FLAGS.log_every_n_steps).replace(
                                        session_config=config)
    summary_dir = os.path.join(FLAGS.model_dir, 'predict')  #预测结果存放目录
    # SSD模型
    ssd_detector = tf.estimator.Estimator(
        model_fn=ssd_model_fn, model_dir=FLAGS.model_dir, config=run_config,
        params={
            'select_threshold': FLAGS.select_threshold,
            'min_size': FLAGS.min_size,
            'nms_threshold': FLAGS.nms_threshold,
            'nms_topk': FLAGS.nms_topk,
            'keep_topk': FLAGS.keep_topk,
            'data_format': FLAGS.data_format,
            'batch_size': FLAGS.batch_size,
            'model_scope': FLAGS.model_scope,
            'save_summary_steps': FLAGS.save_summary_steps,
            'summary_dir': summary_dir,
            'num_classes': FLAGS.num_classes,
            'negative_ratio': FLAGS.negative_ratio,
            'match_threshold': FLAGS.match_threshold,
            'neg_threshold': FLAGS.neg_threshold,
            'weight_decay': FLAGS.weight_decay,
        })
    tensors_to_log = {
        'ce': 'cross_entropy_loss',
        'loc': 'location_loss',
        'loss': 'total_loss',
        'acc': 'cls_accuracy',
    }
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=FLAGS.log_every_n_steps,
                                            formatter=lambda dicts: (', '.join(['%s=%.6f' % (k, v) for k, v in dicts.items()])))
    print('Starting a predict cycle.')
    pred_results = ssd_detector.predict(input_fn=input_pipeline(dataset_pattern='val-*', is_training=False, batch_size=FLAGS.batch_size),
                                    hooks=[logging_hook], checkpoint_path=get_checkpoint())  # 检测
    det_results = list(pred_results)
    for class_ind in range(1, FLAGS.num_classes):
        with open(os.path.join(summary_dir, 'results_{}.txt'.format(class_ind)), 'wt') as f:
            for image_ind, pred in enumerate(det_results):
                filename = pred['filename']
                shape = pred['shape']
                scores = pred['scores_{}'.format(class_ind)]
                bboxes = pred['bboxes_{}'.format(class_ind)]
                bboxes[:, 0] = (bboxes[:, 0] * shape[0]).astype(np.int32, copy=False) + 1
                bboxes[:, 1] = (bboxes[:, 1] * shape[1]).astype(np.int32, copy=False) + 1
                bboxes[:, 2] = (bboxes[:, 2] * shape[0]).astype(np.int32, copy=False) + 1
                bboxes[:, 3] = (bboxes[:, 3] * shape[1]).astype(np.int32, copy=False) + 1
                valid_mask = np.logical_and((bboxes[:, 2] - bboxes[:, 0] > 0), (bboxes[:, 3] - bboxes[:, 1] > 0))  # 有效部分
                for det_ind in range(valid_mask.shape[0]):
                    if not valid_mask[det_ind]:
                        continue
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(filename.decode('utf8')[:-4], scores[det_ind],
                                       bboxes[det_ind, 1], bboxes[det_ind, 0],
                                       bboxes[det_ind, 3], bboxes[det_ind, 2]))


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
