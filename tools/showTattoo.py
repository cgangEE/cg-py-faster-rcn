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

def showTattoo(folder, dataset):

    prototxt = 'test.prototxt'
    caffemodel = 'tattc_voc.caffemodel'
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    dataset = os.path.join(folder, dataset)
    f = open(dataset, 'r') 
    for line in f:
        if random.randint(1, 1000) > 10:
            continue

        line = line[:-1]                 
        imgname = os.path.join(folder, 'images', line)                
        im = cv2.imread(imgname)
        print(imgName)

        scores, boxes = im_detect(net, im)
        print(scores.shape, boxes.shape)
    
    
if __name__ == '__main__':
    showTattoo('flickr', 'flickr10000_group1.txt')
