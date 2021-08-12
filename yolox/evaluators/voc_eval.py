#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Code are based on
# https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
# Copyright (c) Bharath Hariharan.
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import pickle
import re
import xml.etree.ElementTree as ET
import string
import numpy as np
from yolox.data.datasets.my_classes import MY_CLASSES

def parse_rec(results):
     #results ['x1,y1,x2,y2,idss','x1,y1,x2,y2,idss'..] 每张图片的
     objects= []
     if len(results)==0:
         return
     elif len(results)==1:
            obj_struct = {}
            # obj_struct["pose"] = obj.find("pose").text
            # obj_struct["truncated"] = int(obj.find("truncated").text)
            obj_struct["difficult"] = int(0)
            bboxes = re.findall(r"\d+", "{}".format(*results))  # [['data']],变为[[data]]
            obj_struct["name"] = MY_CLASSES[int(bboxes[-1])]
            obj_struct["bbox"] = [
                int(bboxes[0]),  # xmin
                int(bboxes[1]),  # ymin
                int(bboxes[2]),  # xmax
                int(bboxes[3]),  # ymax
            ]
            objects.append(obj_struct)
     else:
         for per_bboxs in results:
            obj_struct = {}
            # obj_struct["pose"] = obj.find("pose").text
            # obj_struct["truncated"] = int(obj.find("truncated").text)
            obj_struct["difficult"] = int(0)
            bboxes = re.findall(r"\d+",per_bboxs)    #[['data']],变为[[data]]
            obj_struct["name"] = MY_CLASSES[int(bboxes[-1])]
            obj_struct["bbox"] = [
                int(bboxes[0]),                 #xmin
                int(bboxes[1]),               # ymin
                int(bboxes[2]),            #xmax
                int(bboxes[3]),       #ymax
            ]
            objects.append(obj_struct)

     return objects                               #     [{"name":  ,"pose":   ,"truncated":   ,"difficult":   ,"bbox":   },{}]每张图片的


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(
    detpath,                  # 每个类别对应的文件夹
    val_txt,                 # annopath 为.xml的绝对路径
    img_paths,             # imagesetfile 为 trainval.txt的绝对路径
    classname,                # 每个类别的名字
    cachedir,                 # 将要保存的路径
    ovthresh=0.5,
    use_07_metric=False,
):
    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, "annots.pkl")
    # read list of images
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        with open(val_txt,'r') as f:
            lines = f.readlines()
            for i,line  in enumerate(lines):
                line = line.rstrip()
                imagename = line.split(' ')[0]
                results = line.split(' ')[1:]
                recs[imagename] = parse_rec(results)    #    recs[imagename] = [{"name":  ,"pose":   ,"truncated":   ,"difficult":   ,"bbox":   }]
            if i % 100 == 0:
                print("Reading annotation for {:d}/{:d}".format(i + 1, len(lines)))
        # save
        print("Saving cached annotations to {:s}".format(cachefile))
        with open(cachefile, "wb") as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, "rb") as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    imagenames = [per_imgname.strip() for per_imgname in img_paths]
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == classname]       #  R  {}
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()

    if len(lines) == 0:
        return 0, 0, 0

    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[-1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[1:-1]] for x in splitlines])
    # print(BB)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

        # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap

# if __name__=='__main__':
#     result = ['0,0,4,5,2','4,5,6,7,8','3,5,45,45,5']
#     out = parse_rec(result)
#     print(out)