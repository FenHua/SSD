# -*- coding: utf-8 -*-
# Description :SSD主函数.

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg

from ssd300_vgg import SSD
from utils import preprocess_image,process_bboxes
from drawbox import plt_bboxes

def main():
    # 【1】搭建网络-->解码网络输出-->设置图片的占位节点
    ssd_net = SSD()  # 建立SSD模型
    classes, scores, bboxes = ssd_net.detections()  # 网络中检测函数，返回类别，置信度，回归框位置
    images = ssd_net.images()  # 给一个图片空间(开辟一个大小)
    # 【2】用检查点文件恢复模型
    sess = tf.Session()
    ckpt_filename = './SSD_model/ssd_vgg_300_weights.ckpt'  # 检查点文件
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_filename)  # 参数恢复
    # 【3】预处理图片-->处理预测边界框bboxes
    img = cv2.imread('./SSD_data/test0.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_prepocessed = preprocess_image(img)  # 预处理图片
    rclasses, rscores, rbboxes = sess.run([classes, scores, bboxes], feed_dict={images: img_prepocessed})  # 检测
    rclasses, rscores, rbboxes = process_bboxes(rclasses, rscores, rbboxes)  # 处理预测边界框(剪掉多余的边界，按照置信度保留top-k个框，调整框大小)
    # 【4】可视化最终的检测结果
    plt_bboxes(img, rclasses, rscores, rbboxes)
    print('SSD detection has done!')

if __name__ == '__main__':
    main()