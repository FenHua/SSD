# -*- coding: utf-8 -*-
# Description :SSD的先验框anchor设置

import math
import numpy as np


def ssd_size_bounds_to_values(size_bounds, n_feat_layers, img_shape=(300, 300)):
    """
    根据相关的边界信息计算先验anchor的大小，函数返回每一种尺度下的绝对大小
    """
    assert img_shape[0] == img_shape[1]
    img_size = img_shape[0]
    min_ratio = int(size_bounds[0] * 100)
    max_ratio = int(size_bounds[1] * 100)
    step = int(math.floor((max_ratio - min_ratio) / (n_feat_layers - 2)))
    sizes = [[img_size * size_bounds[0] / 2, img_size * size_bounds[0]]]  # 从最小的尺度开始
    for ratio in range(min_ratio, max_ratio + 1, step):
        sizes.append((img_size * ratio / 100.,
                      img_size * (ratio + step) / 100.))
    return sizes

def ssd_anchor_one_layer(img_shape, feat_shape, sizes, ratios, step, offset=0.5, dtype=np.float32):
    """
    计算每一特征层的默认anchor框，决定相关位置的中心，宽，高。
    feat_shape: 特征大小信息; size: 绝对参考大小; ratios: 不同特征层下的比率;
    img_shape: 图片大小; ofset: 网格坐标
    返回:y, x, h, w: 网格坐标x，y 和长宽信息
    """
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(dtype) + offset) * step / img_shape[0]
    x = (x.astype(dtype) + offset) * step / img_shape[1]
    # 扩充一维，便于使用广播机制
    y = np.expand_dims(y, axis=-1)  # [size, size, 1]
    x = np.expand_dims(x, axis=-1)  # [size, size, 1]
    # 计算相关的长和宽
    num_anchors = len(sizes) + len(ratios)
    h = np.zeros((num_anchors, ), dtype=dtype)  # [n_anchors]
    w = np.zeros((num_anchors, ), dtype=dtype)  # [n_anchors]
    # 添加比率为1的anchor boxes
    h[0] = sizes[0] / img_shape[0]
    w[0] = sizes[0] / img_shape[1]
    di = 1
    if len(sizes) > 1:
        h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
        w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
        di += 1
    for i, r in enumerate(ratios):
        h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r)
        w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r)
    return y, x, h, w


def ssd_anchors_all_layers(img_shape, layers_shape, anchor_sizes, anchor_ratios, anchor_steps, offset=0.5, dtype=np.float32):
    # 计算每一特征层的anchor
    layers_anchors = []
    for i, s in enumerate(layers_shape):
        anchor_bboxes = ssd_anchor_one_layer(img_shape, s, anchor_sizes[i], anchor_ratios[i], anchor_steps[i], offset=offset, dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors