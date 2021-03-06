# -*- coding: utf-8 -*-
# SSD功能函数：预处理图片、处理/筛选预测边界框.

import cv2
import numpy as np


# 图片均值化为0
def whiten_image(image, means=(123., 117., 104.)):
    if image.ndim != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.shape[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    mean = np.array(means, dtype=image.dtype)
    image = image - mean  # 从每个图像通道中减去给定的均值
    return image


def resize_image(image, size=(300, 300)):
    return cv2.resize(image, size)


# 预处理图片preprocess_image
def preprocess_image(image):
    image_cp = np.copy(image).astype(np.float32)
    image_whitened = whiten_image(image_cp)  # 图片均值化为0
    image_resized = resize_image(image_whitened)  # resize
    image_expanded = np.expand_dims(image_resized, axis=0)  # 增加batch_size这一维度[batchsize,width,height]
    return image_expanded


# cut the box:将边界框超出整张图片(0,0)—(300,300)的部分cut掉
def bboxes_clip(bbox_ref, bboxes):
    bboxes = np.copy(bboxes)
    bboxes = np.transpose(bboxes)
    bbox_ref = np.transpose(bbox_ref)
    bboxes[0] = np.maximum(bboxes[0], bbox_ref[0])  # xmin
    bboxes[1] = np.maximum(bboxes[1], bbox_ref[1])  # ymin
    bboxes[2] = np.minimum(bboxes[2], bbox_ref[2])  # xmax
    bboxes[3] = np.minimum(bboxes[3], bbox_ref[3])  # ymax
    bboxes = np.transpose(bboxes)
    return bboxes


# 按类别置信度scores降序，对边界框进行排序并仅保留top_k=400
def bboxes_sort(classes, scores, bboxes, top_k=400):
    idxes = np.argsort(-scores)
    classes = classes[idxes][:top_k]
    scores = scores[idxes][:top_k]
    bboxes = bboxes[idxes][:top_k]
    return classes, scores, bboxes


# 计算IOU
def bboxes_iou(bboxes1, bboxes2):
    bboxes1 = np.transpose(bboxes1)
    bboxes2 = np.transpose(bboxes2)
    # 计算两个box的交集：交集左上角的点取两个box的max，交集右下角的点取两个box的min
    int_ymin = np.maximum(bboxes1[0], bboxes2[0])
    int_xmin = np.maximum(bboxes1[1], bboxes2[1])
    int_ymax = np.minimum(bboxes1[2], bboxes2[2])
    int_xmax = np.minimum(bboxes1[3], bboxes2[3])
    # 计算两个box交集的wh：如果两个box没有交集，那么wh为0(按照计算方式wh为负数，跟0比较取最大值)
    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)
    # 计算IOU
    int_vol = int_h * int_w  # 交集面积
    vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])  # bboxes1面积
    vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])  # bboxes2面积
    iou = int_vol / (vol1 + vol2 - int_vol) # IOU=交集/并集
    return iou


# NMS(tf.image.non_max_suppression(boxes, scores,self.max_output_size, self.iou_threshold))
def bboxes_nms(classes, scores, bboxes, nms_threshold=0.5):
    keep_bboxes = np.ones(scores.shape, dtype=np.bool)
    for i in range(scores.size-1):
        if keep_bboxes[i]:
            overlap = bboxes_iou(bboxes[i], bboxes[(i+1):])  # 计算IOU
            keep_overlap = np.logical_or(overlap < nms_threshold, classes[(i+1):] != classes[i])  # 条件：小于阈值并且相邻类别不同
            keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):], keep_overlap)
    idxes = np.where(keep_bboxes)
    return classes[idxes], scores[idxes], bboxes[idxes]

# 根据先验框anchor调整预测边界框的大小
def bboxes_resize(bbox_ref, bboxes):
    """
    根据anchor数据重新给定bounding boxes大小
    assuming that the latter is [0, 0, 1, 1] after transform.
    """
    bboxes = np.copy(bboxes)
    # Translate.
    bboxes[:, 0] -= bbox_ref[0]
    bboxes[:, 1] -= bbox_ref[1]
    bboxes[:, 2] -= bbox_ref[0]
    bboxes[:, 3] -= bbox_ref[1]
    # Resize.
    resize = [bbox_ref[2] - bbox_ref[0], bbox_ref[3] - bbox_ref[1]]
    bboxes[:, 0] /= resize[0]
    bboxes[:, 1] /= resize[1]
    bboxes[:, 2] /= resize[0]
    bboxes[:, 3] /= resize[1]
    return bboxes

# 处理预测边界框 process_bboxes
def process_bboxes(rclasses, rscores, rbboxes, rbbox_img = (0.0, 0.0, 1.0, 1.0),
                   top_k=400, nms_threshold=0.5):
    rbboxes = bboxes_clip(rbbox_img, rbboxes)  # 将边界框超出整张图片(0,0)—(300,300)的部分cut掉
    rclasses, rscores, rbboxes = bboxes_sort(rclasses, rscores, rbboxes, top_k)  # 按类别scores降序，对边界框进行排序保留top_k=400
    rclasses, rscores, rbboxes = bboxes_nms(rclasses, rscores, rbboxes, nms_threshold)  # 计算IOU-->NMS
    rbboxes = bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes