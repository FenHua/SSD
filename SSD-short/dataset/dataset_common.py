from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

VOC_LABELS = {
    'none': (0, 'Background'),
    'aeroplane': (1, 'Vehicle'),
    'bicycle': (2, 'Vehicle'),
    'bird': (3, 'Animal'),
    'boat': (4, 'Vehicle'),
    'bottle': (5, 'Indoor'),
    'bus': (6, 'Vehicle'),
    'car': (7, 'Vehicle'),
    'cat': (8, 'Animal'),
    'chair': (9, 'Indoor'),
    'cow': (10, 'Animal'),
    'diningtable': (11, 'Indoor'),
    'dog': (12, 'Animal'),
    'horse': (13, 'Animal'),
    'motorbike': (14, 'Vehicle'),
    'person': (15, 'Person'),
    'pottedplant': (16, 'Indoor'),
    'sheep': (17, 'Animal'),
    'sofa': (18, 'Indoor'),
    'train': (19, 'Vehicle'),
    'tvmonitor': (20, 'Indoor'),
}

COCO_LABELS = {
    "bench":  (14, 'outdoor') ,
    "skateboard":  (37, 'sports') ,
    "toothbrush":  (80, 'indoor') ,
    "person":  (1, 'person') ,
    "donut":  (55, 'food') ,
    "none":  (0, 'background') ,
    "refrigerator":  (73, 'appliance') ,
    "horse":  (18, 'animal') ,
    "elephant":  (21, 'animal') ,
    "book":  (74, 'indoor') ,
    "car":  (3, 'vehicle') ,
    "keyboard":  (67, 'electronic') ,
    "cow":  (20, 'animal') ,
    "microwave":  (69, 'appliance') ,
    "traffic light":  (10, 'outdoor') ,
    "tie":  (28, 'accessory') ,
    "dining table":  (61, 'furniture') ,
    "toaster":  (71, 'appliance') ,
    "baseball glove":  (36, 'sports') ,
    "giraffe":  (24, 'animal') ,
    "cake":  (56, 'food') ,
    "handbag":  (27, 'accessory') ,
    "scissors":  (77, 'indoor') ,
    "bowl":  (46, 'kitchen') ,
    "couch":  (58, 'furniture') ,
    "chair":  (57, 'furniture') ,
    "boat":  (9, 'vehicle') ,
    "hair drier":  (79, 'indoor') ,
    "airplane":  (5, 'vehicle') ,
    "pizza":  (54, 'food') ,
    "backpack":  (25, 'accessory') ,
    "kite":  (34, 'sports') ,
    "sheep":  (19, 'animal') ,
    "umbrella":  (26, 'accessory') ,
    "stop sign":  (12, 'outdoor') ,
    "truck":  (8, 'vehicle') ,
    "skis":  (31, 'sports') ,
    "sandwich":  (49, 'food') ,
    "broccoli":  (51, 'food') ,
    "wine glass":  (41, 'kitchen') ,
    "surfboard":  (38, 'sports') ,
    "sports ball":  (33, 'sports') ,
    "cell phone":  (68, 'electronic') ,
    "dog":  (17, 'animal') ,
    "bed":  (60, 'furniture') ,
    "toilet":  (62, 'furniture') ,
    "fire hydrant":  (11, 'outdoor') ,
    "oven":  (70, 'appliance') ,
    "zebra":  (23, 'animal') ,
    "tv":  (63, 'electronic') ,
    "potted plant":  (59, 'furniture') ,
    "parking meter":  (13, 'outdoor') ,
    "spoon":  (45, 'kitchen') ,
    "bus":  (6, 'vehicle') ,
    "laptop":  (64, 'electronic') ,
    "cup":  (42, 'kitchen') ,
    "bird":  (15, 'animal') ,
    "sink":  (72, 'appliance') ,
    "remote":  (66, 'electronic') ,
    "bicycle":  (2, 'vehicle') ,
    "tennis racket":  (39, 'sports') ,
    "baseball bat":  (35, 'sports') ,
    "cat":  (16, 'animal') ,
    "fork":  (43, 'kitchen') ,
    "suitcase":  (29, 'accessory') ,
    "snowboard":  (32, 'sports') ,
    "clock":  (75, 'indoor') ,
    "apple":  (48, 'food') ,
    "mouse":  (65, 'electronic') ,
    "bottle":  (40, 'kitchen') ,
    "frisbee":  (30, 'sports') ,
    "carrot":  (52, 'food') ,
    "bear":  (22, 'animal') ,
    "hot dog":  (53, 'food') ,
    "teddy bear":  (78, 'indoor') ,
    "knife":  (44, 'kitchen') ,
    "train":  (7, 'vehicle') ,
    "vase":  (76, 'indoor') ,
    "banana":  (47, 'food') ,
    "motorcycle":  (4, 'vehicle') ,
    "orange":  (50, 'food')
  }

# use dataset_inspect.py to get these summary
data_splits_num = {
    'train': 22136,
    'val': 4952,
}

def slim_get_batch(num_classes, batch_size, split_name, file_pattern, num_readers, num_preprocessing_threads, image_preprocessing_fn, anchor_encoder, num_epochs=None, is_training=True):
    """
    根据指示从VOC数据集中得到数据集元组
    输入:
      num_classes: 数据集中总共的类别数
      batch_size: batch_size大小
      split_name: 'train' of 'val'.
      file_pattern: 当进行数据集匹配时，使用文件的模式
      num_readers: 最大的读取tf数据的线程数.
      num_preprocessing_threads: 最大的用于预处理的线程数
      image_preprocessing_fn: 用于数据增强的函数
      anchor_encoder: 编码anchor的函数
      num_epochs: 总共迭代此数据集的次数
      is_training: 是否处于训练阶段.
    返回:
      batch大小的数据[image, shape, loc_targets, cls_targets, match_scores].
    """
    if split_name not in data_splits_num:
        raise ValueError('split name %s was not recognized.' % split_name)
    # TFRecords形式VOC数据
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64),
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'filename': slim.tfexample_decoder.Tensor('image/filename'),
        'shape': slim.tfexample_decoder.Tensor('image/shape'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
                ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
        'object/difficult': slim.tfexample_decoder.Tensor('image/object/bbox/difficult'),
        'object/truncated': slim.tfexample_decoder.Tensor('image/object/bbox/truncated'),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    labels_to_names = {}
    for name, pair in VOC_LABELS.items():
        labels_to_names[pair[0]] = name

    dataset = slim.dataset.Dataset(
                data_sources=file_pattern,
                reader=tf.TFRecordReader,
                decoder=decoder,
                num_samples=data_splits_num[split_name],
                items_to_descriptions=None,
                num_classes=num_classes,
                labels_to_names=labels_to_names)

    with tf.name_scope('dataset_data_provider'):
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=num_readers,
            common_queue_capacity=32 * batch_size,
            common_queue_min=8 * batch_size,
            shuffle=is_training,
            num_epochs=num_epochs)

    [org_image, filename, shape, glabels_raw, gbboxes_raw, isdifficult] = provider.get(['image', 'filename', 'shape',
                                                                     'object/label',
                                                                     'object/bbox',
                                                                     'object/difficult'])
    if is_training:
        # if all is difficult, then keep the first one
        isdifficult_mask =tf.cond(tf.count_nonzero(isdifficult, dtype=tf.int32) < tf.shape(isdifficult)[0],
                                lambda : isdifficult < tf.ones_like(isdifficult),
                                lambda : tf.one_hot(0, tf.shape(isdifficult)[0], on_value=True, off_value=False, dtype=tf.bool))

        glabels_raw = tf.boolean_mask(glabels_raw, isdifficult_mask)
        gbboxes_raw = tf.boolean_mask(gbboxes_raw, isdifficult_mask)
    # Pre-processing image, labels and bboxes.
    if is_training:
        image, glabels, gbboxes = image_preprocessing_fn(org_image, glabels_raw, gbboxes_raw)
    else:
        image = image_preprocessing_fn(org_image, glabels_raw, gbboxes_raw)
        glabels, gbboxes = glabels_raw, gbboxes_raw
    gt_targets, gt_labels, gt_scores = anchor_encoder(glabels, gbboxes)
    return tf.train.batch([image, filename, shape, gt_targets, gt_labels, gt_scores],
                    dynamic_pad=False,
                    batch_size=batch_size,
                    allow_smaller_final_batch=(not is_training),
                    num_threads=num_preprocessing_threads,
                    capacity=64 * batch_size)