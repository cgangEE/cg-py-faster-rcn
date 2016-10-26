#!/usr/bin/env python


import os, sys, cv2
os.environ['GLOG_minloglevel'] = '2' 

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import argparse
import glob
import caffe, random


CLASSES = ('__background__', # always index 0, total 22
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor',
           'tattoo')

def vis_detections(im, class_name, dets, idx, thresh=0.5):
    """Draw detected bounding boxes."""
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    for i in range(len(dets)):
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name, thresh), fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.savefig('tattoo_detected_' + str(idx) + 'png')

def tattoo_detection(net, image_name, tattooSet, idx):
    global totTattoo
    global errTattoo
    global nonTattoo
    global errNonTattoo

    im_in = cv2.imread(os.path.join(folder, 'images', image_name))

    if im_in is None:
        print('cannot open %s for read' % image_name )
        exit(-1)

    rows,cols = im_in.shape[:2]

    scale=1.0
    CONF_THRESH = 0.3
    NMS_THRESH  = 0.3
    longdim = 500

    if rows >= cols:
        scale = float(longdim) / float(rows)
        im = cv2.resize( im_in, (int(0.5 + float(cols)*scale), longdim) )
    else:
        scale = float(longdim) / float(cols)
        im = cv2.resize( im_in, (longdim, int(0.5 + float(rows)*scale)) )

    scores, boxes = im_detect(net, im)
    max_scores = scores.max(axis=0)

    cls_ind = 21
    cls = 'tattoo'

    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]

    inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
    dets = dets[inds]

    if image_name in tattooSet:
        totTattoo += 1
        if len(dets) == 0:
            errTattoo += 1
    else:
        nonTattoo += 1
        if len(dets) >0:
            errNonTattoo += 1
            
    if (len(dets>0)):
        vis_detections(im, cls, dets, idx, thresh=CONF_THRESH)


def readGt(gtName):
    f = open(gtName, 'r')       
    ret = []
    for line in f:
        if line[0] == 't':
            ret.append(line[7:-1])

    return set(ret) 

if __name__ == '__main__':
    
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    cfg.TEST.BBOX_REG = True #False
    cfg.TEST.NMS=0.3 #0.50
    cfg.TEST.RPN_NMS_THRESH=0.50
    cfg.TEST.RPN_POST_NMS_TOP_N = 5000 #20000

    prototxt = 'test.prototxt'
    caffemodel = 'tattc_voc.caffemodel'

    caffe.set_mode_gpu()
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    folder = 'flickr'
    dataset = 'flickr10000_group1.txt'
    dataset = os.path.join(folder, dataset)

    tattooSet = readGt(os.path.join('flickr', 'ground_truth.txt'))

    totTattoo = 0
    errTattoo = 0
    nonTattoo = 0
    errNonTattoo = 0

    f = open(dataset, 'r') 
    idx = 0
    for line in f:
        line = line[:-1]                 
        tattoo_detection(net, line, tattooSet, idx)
        idx += 1
        print(totTattoo, errTattoo, nonTattoo, errNonTattoo)
        if idx == 5:
            break

    print(totTattoo, errTattoo, nonTattoo, errNonTattoo)
    plt.show()

