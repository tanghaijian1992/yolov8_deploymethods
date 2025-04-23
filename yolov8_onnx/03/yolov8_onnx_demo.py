#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import os
import sys
import os.path as osp
import cv2
import torch
import numpy as np
import onnxruntime as ort
from math import exp

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


CLASSES = ['plate_base']


class_num = len(CLASSES)
head_num = 3
nms_thresh = 0.45
object_thresh = 0.3

strides = [8, 16, 32]
map_size = [[80, 80], [40, 40], [20, 20]]

input_height = 640
input_width = 640

meshgrid = []

anchors = int(input_height / strides[0] * input_width / strides[0] + input_height / strides[1] * input_width / strides[1] + input_height / strides[2] * input_width / strides[2])


class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


def GenerateMeshgrid():
    for index in range(head_num):
        for i in range(map_size[index][0]):
            for j in range(map_size[index][1]):
                meshgrid.append(j + 0.5)
                meshgrid.append(i + 0.5)


def IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    innerWidth = xmax - xmin
    innerHeight = ymax - ymin

    innerWidth = innerWidth if innerWidth > 0 else 0
    innerHeight = innerHeight if innerHeight > 0 else 0

    innerArea = innerWidth * innerHeight

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    total = area1 + area2 - innerArea

    return innerArea / total


def NMS(detectResult):
    predBoxs = []

    sort_detectboxs = sorted(detectResult, key=lambda x: x.score, reverse=True)

    for i in range(len(sort_detectboxs)):
        xmin1 = sort_detectboxs[i].xmin
        ymin1 = sort_detectboxs[i].ymin
        xmax1 = sort_detectboxs[i].xmax
        ymax1 = sort_detectboxs[i].ymax
        classId = sort_detectboxs[i].classId

        if sort_detectboxs[i].classId != -1:
            predBoxs.append(sort_detectboxs[i])
            for j in range(i + 1, len(sort_detectboxs), 1):
                if classId == sort_detectboxs[j].classId:
                    xmin2 = sort_detectboxs[j].xmin
                    ymin2 = sort_detectboxs[j].ymin
                    xmax2 = sort_detectboxs[j].xmax
                    ymax2 = sort_detectboxs[j].ymax
                    iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2)
                    if iou > nms_thresh:
                        sort_detectboxs[j].classId = -1
    return predBoxs



def postprocess(outputs, image_h, image_w):
    print('postprocess ... ')

    scale_h = image_h / input_height
    scale_w = image_w / input_width
    
    detectResult = []
    output_cls = outputs[0]
    output_reg = outputs[1]

    grid_index = -2
    
    for i in range(0, anchors):
        cls_index = 0
        cls_vlaue = -1

        grid_index += 2
        
        for cl in range(class_num):
            val = output_cls[i + cl * anchors]
            if val > cls_vlaue:
                cls_vlaue = val
                cls_index = cl
                
        if cls_vlaue > object_thresh:
            cx = output_reg[i + 0 * anchors]
            cy = output_reg[i + 1 * anchors]
            cw = output_reg[i + 2 * anchors]
            ch = output_reg[i + 3 * anchors]

            if i < map_size[0][0] * map_size[0][1]:
                index = 0
            elif i < map_size[0][0] * map_size[0][1] + map_size[1][0] * map_size[1][1]:
                index = 1
            else:
                index = 2

            xmin = (meshgrid[grid_index + 0] - cx) * strides[index]
            ymin = (meshgrid[grid_index + 1] - cy) * strides[index]
            xmax = (meshgrid[grid_index + 0] + cw) * strides[index]
            ymax = (meshgrid[grid_index + 1] + ch) * strides[index]
            
            xmin *= scale_w
            ymin *= scale_h
            xmax *= scale_w
            ymax *= scale_h

            box = DetectBox(cls_index, cls_vlaue, xmin, ymin, xmax, ymax)
            detectResult.append(box)
    # NMS
    print('detectResult:', len(detectResult))
    predBox = NMS(detectResult)

    return predBox


def precess_image(img_src, resize_w, resize_h):
    image = cv2.resize(img_src, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image /= 255.0

    return image


def detect(img_path):

    orig = cv2.imread(img_path)
    img_h, img_w = orig.shape[:2]
    image = precess_image(orig, input_height, input_width)

    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)

    ort_session = ort.InferenceSession('./yolov8_plate.onnx')
    pred_results = (ort_session.run(None, {'images': image}))

    outputs = []
    for i in range(len(pred_results)):
        print(pred_results[i].shape)
        outputs.append(pred_results[i].reshape(-1))
    predbox = postprocess(outputs, img_h, img_w)

    print('obj num is :', len(predbox))

    for i in range(len(predbox)):
        xmin = int(predbox[i].xmin)
        ymin = int(predbox[i].ymin)
        xmax = int(predbox[i].xmax)
        ymax = int(predbox[i].ymax)
        classId = predbox[i].classId
        score = predbox[i].score

        cv2.rectangle(orig, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        ptext = (xmin, ymin)
        title = CLASSES[classId] + "%.2f" % score
        cv2.putText(orig, title, ptext, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite('./test_onnx_result.jpg', orig)


if __name__ == '__main__':
    print('This is main ....')
    GenerateMeshgrid()
    img_path = './test.jpg'
    detect(img_path)