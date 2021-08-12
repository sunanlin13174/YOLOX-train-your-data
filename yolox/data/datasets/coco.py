#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import pickle
import cv2
import numpy as np
from .my_classes import MY_CLASSES
from .datasets_wrapper import Dataset
from yolox.evaluators.voc_eval import voc_eval

class COCODataset(Dataset):
    """
    COCO dataset class.
    """

    def __init__(
        self,
        label_txt = '/home/meprint/sunanlin_folder/deepfashion2/labels/train2017',
        img_size=(416, 416),
        preproc=None,
        num_classes=38
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__(img_size)
        self.img_path = []
         ########修改了 self.coco的参数路径信息。
        #格式：  img_path  x1,y1,x2,y2,idss  x1,y1,x2,y2,idss
        #返回的res格式为 [[x1,y1,x2,y2,c],[x1,y1,x2,y2,c]....]
        self.result =[]
        self.label_txt = label_txt
        self.class_ids = [i for i in range(int(num_classes))]           #对应meprint_class.py的38个类
        # print(label_txt)
        with open(self.label_txt,'r') as f:
             for ix,line  in enumerate(f):
                    line = line.strip('\n')
                    files = line.split(' ')
                    img_path= files[0]
                    self.img_path.append(img_path)
                    bbox_cls = files[1:]
                    if bbox_cls[-1]=="":
                        continue
                    self.bbox_=[]
                    for per_bbox_cls in bbox_cls:                  # bbox_cls :[str::bbox1 bbox2 ...]
                        per_bbox_cls = per_bbox_cls.strip('')
                        x1 = np.float(per_bbox_cls.split(',')[0])
                        y1 = np.float(per_bbox_cls.split(',')[1])
                        x2 = np.float(per_bbox_cls.split(',')[2])
                        y2 = np.float(per_bbox_cls.split(',')[3])
                        c = int(per_bbox_cls.split(',')[4])
                        assert x1>=0 and x2>0 and y1>=0 and y2>0
                        self.bbox_.append((x1,y1,x2,y2,c))         #返回每张图片的  float类型坐标，去掉str   bbox_cls :[float::bbox1 bbox2 ...]
                    self.result.append(self.bbox_)

        print('列表中一共有{}张图片'.format(len(self.img_path)))
        self.data = self.result   #[[x1,y1,x2,y2,c],[]] 每一张图片[    [[x1,x2,x3,x4,c],[x1,x2,x3,x4,c]],[[  ]]                 ]

        print('我们的标签长度为：',len(self.data))
        self.img_size = img_size
        self.preproc = preproc

    def load_anno(self, index):
        res = self.data[index]
        return res       # [[xmin, ymin, xmax, ymax, label_ind], ... ]每张图片对应的多个目标bbox_

    def __len__(self):
        return len(self.img_path)

    def pull_item(self, index):
        id_ = self.img_path[index]
        # print(id_)
        img = cv2.imread(id_, cv2.IMREAD_COLOR)
        assert img is not None
        height, width, _ = img.shape
        img_info = (width, height)
        bbox_ = np.array(self.load_anno(index))
        return img, bbox_, img_info, index

    @Dataset.resize_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            img_id (int): same as the input index. Used for evaluation.
        """
        img, res, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, res, self.input_dim)
        return img, target, img_info, img_id
    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        self._write_voc_results_file(all_boxes)
        IouTh = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
        mAPs = []
        for iou in IouTh:
            mAP = self._do_python_eval(output_dir, iou)
            mAPs.append(mAP)

        print("--------------------------------------------------------------")
        print("map_5095:", np.mean(mAPs))
        print("map_50:", mAPs[0])
        print("--------------------------------------------------------------")
        return np.mean(mAPs), mAPs[0]

    def _get_voc_results_file_template(self):
        filename = "comp4_det_test" + "_{:s}.txt"
        filedir = "eval_results"
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(MY_CLASSES):
            cls_ind = cls_ind
            if cls == "__background__":
                continue
            print("Writing {} VOC results file".format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, "wt") as f:
                for im_ind, index in enumerate(self.img_path):             #index 为（rootpath,img_name）
                    index = index.strip()                                  #index 为img_name
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        f.write(
                            "{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n".format(
                                index,
                                dets[k, 0] + 1,
                                dets[k, 1] + 1,
                                dets[k, 2] + 1,
                                dets[k, 3] + 1,
                                dets[k, -1],
                            )
                        )

    def _do_python_eval(self, output_dir="output", iou=0.5):
               #VOC数据集根目录
                                         # name 为trainval

        # imagesetfile = os.path.join(rootpath, "ImageSets", "Main", name + ".txt")            #imagesetfile 为 trainval.txt的绝对路径

        cachedir = "result_cache"
        if not os.path.exists(cachedir):
            os.makedirs(cachedir)
        aps = []
        # The PASCAL VOC metric changed in 2010
        # use_07_metric = True if int(self._year) < 2010 else False
        use_07_metric = False
        print("Eval IoU : {:.2f}".format(iou))
        if output_dir is not None and not os.path.isdir(output_dir):                        #建立存储的文件夹
            os.mkdir(output_dir)
        for i, cls in enumerate(MY_CLASSES):

            if cls == "__background__":
                continue

            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename,         #每个类别对应的文件夹
                self.label_txt,          # annopath 为.xml的绝对路径
                self.img_path,        #imagesetfile 为 trainval.txt的绝对路径
                cls,                   #每个类别的名字
                cachedir,              #将要保存的路径
                ovthresh=iou,
                use_07_metric=use_07_metric,
            )
            aps += [ap]
            if iou == 0.5:
                print("AP for {} = {:.4f}".format(cls, ap))
            if output_dir is not None:
                with open(os.path.join(output_dir, cls + "_pr.pkl"), "wb") as f:
                    pickle.dump({"rec": rec, "prec": prec, "ap": ap}, f)
        if iou == 0.5:
            print("Mean AP = {:.4f}".format(np.mean(aps)))
            print("~~~~~~~~")
            print("Results:")
            for ap in aps:
                print("{:.3f}".format(ap))
            print("{:.3f}".format(np.mean(aps)))
            print("~~~~~~~~")
            print("")
            print("--------------------------------------------------------------")
            print("Results computed with the **unofficial** Python eval code.")
            print("Results should be very close to the official MATLAB eval code.")
            print("Recompute with `./tools/reval.py --matlab ...` for your paper.")
            print("-- Thanks, The Management")
            print("--------------------------------------------------------------")

        return np.mean(aps)
