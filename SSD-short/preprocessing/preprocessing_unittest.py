from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
from scipy.misc import imread, imsave, imshow, imresize
import numpy as np
import sys;
sys.path.insert(0, ".")
from utility import draw_toolbox
import ssd_preprocessing
slim = tf.contrib.slim


def save_image_with_bbox(image, labels_, scores_, bboxes_):
    # 在图片上标记bounding boxes
    if not hasattr(save_image_with_bbox, "counter"):
        save_image_with_bbox.counter = 0  # 如果不存在，就初始化一个
    save_image_with_bbox.counter += 1
    img_to_draw = np.copy(image)
    img_to_draw = draw_toolbox.bboxes_draw_on_img(img_to_draw, labels_, scores_, bboxes_, thickness=2)  # 画框
    imsave(os.path.join('./debug/{}.jpg').format(save_image_with_bbox.counter), img_to_draw)
    return save_image_with_bbox.counter


def slim_get_split(file_pattern='{}_????'):
    # VOC tf数据特征
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
    }  # 特征关键字
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
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)  # 解码器
    dataset = slim.dataset.Dataset(
                data_sources=file_pattern,
                reader=tf.TFRecordReader,
                decoder=decoder,
                num_samples=100,
                items_to_descriptions=None,
                num_classes=21,
                labels_to_names=None)
    with tf.name_scope('dataset_data_provider'):
        provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    num_readers=2,
                    common_queue_capacity=32,
                    common_queue_min=8,
                    shuffle=True,
                    num_epochs=1)
    [org_image, filename, shape, glabels_raw, gbboxes_raw, isdifficult] = provider.get(['image', 'filename', 'shape',
                                                                         'object/label',
                                                                         'object/bbox',
                                                                         'object/difficult'])
    #  获取对应的图片和label以及目标框的信息
    image, glabels, gbboxes = ssd_preprocessing.preprocess_image(org_image, glabels_raw, gbboxes_raw, [300, 300], is_training=True, data_format='channels_first', output_rgb=True)
    image = tf.transpose(image, perm=(1, 2, 0))
    save_image_op = tf.py_func(save_image_with_bbox,
                            [ssd_preprocessing.unwhiten_image(image),
                            tf.clip_by_value(glabels, 0, tf.int64.max),
                            tf.ones_like(glabels),
                            gbboxes],
                            tf.int64, stateful=True)
    return save_image_op

if __name__ == '__main__':
    save_image_op = slim_get_split('../dataset/tfrecords/*')
    # 创建tensorflow graph等信息
    init_op = tf.group([tf.local_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])
    sess = tf.Session()  # 创建一个session用于运行graph上的一系列操作
    sess.run(init_op)  # 初始化变量 (像每个epoch的counter)
    coord = tf.train.Coordinator()  # 输入队列线程
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        while not coord.should_stop():
            print(sess.run(save_image_op))  # 输出训练步
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()  # 完成，停止线程工作
    # 等待后续工作完成
    coord.join(threads)
    sess.close()
