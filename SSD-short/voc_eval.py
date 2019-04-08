from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import pickle
from dataset import dataset_common
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
'''
# 文件夹形式
VOC2007TEST
    Annotations
    ...
    ImageSets
'''
dataset_path = './VOC/VOC2007TEST'  # VOC文件夹路经
pred_path = './logs/predict'  # 预测结果文件夹路经
pred_file = 'results_{}.txt'  # from 1-num_classes
output_path = './logs/predict/eval_output'
cache_path = './logs/predict/eval_cache'
anno_files = 'Annotations/{}.xml'
all_images_file = 'ImageSets/Main/test.txt'


def parse_rec(filename):
    # 解析 PASCAL VOC xml 文件
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)
    return objects


def do_python_eval(use_07=True):
    # 对当前算法效果进行验证
    aps = []
    use_07_metric = use_07  # VOC数据集2010有所变化
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))  # 判断当前是否是VOC2007
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    for cls_name, cls_pair in dataset_common.VOC_LABELS.items():
        if 'none' in cls_name:
            continue
        cls_id = cls_pair[0]
        filename = os.path.join(pred_path, pred_file.format(cls_id))
        rec, prec, ap = voc_eval(filename, os.path.join(dataset_path, anno_files),
                os.path.join(dataset_path, all_images_file), cls_name, cache_path,
                ovthresh=0.5, use_07_metric=use_07_metric)  # 进行数据集验证
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls_name, ap))
        with open(os.path.join(output_path, cls_name + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')


def voc_ap(rec, prec, use_07_metric=True):
    # 用precision 和 recall计算VOC的AP
    if use_07_metric:
        # 采用11点矩阵方式计算AP
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        #  AP计算
        # 首先将上限值添加到最后
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,annopath,imagesetfile,classname,cachedir,ovthresh=0.5,use_07_metric=True):
    """
    PASCAL VOC 数据集验证
    detpath: 检测路经
    annopath: 标注数据的路经，xml文件
    imagesetfile: Text文件包含一系列图片
    classname: 类别名称
    cachedir: 存储标注数据的目录
    [ovthresh]: IOU阈值
    [use_07_metric]: 是否使用VOC07数据集，此项决定着是否采用11点法
    """
    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # 读取图片列表
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    if not os.path.isfile(cachefile):
        # 加载标注数据
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(i + 1, len(imagenames)))
        # 存储
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # 加载
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)
    # 从当前类别中获取真实目标
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}
    # 读取检测结果
    with open(detpath, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:
        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
        # 根据置信度排序
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]
        # 获取每张图片并计算TP和FP
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # 计算IOU
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.
        # 计算precision和recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)  # 防止除数为0
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.
    return rec, prec, ap

if __name__ == '__main__':
    do_python_eval()
