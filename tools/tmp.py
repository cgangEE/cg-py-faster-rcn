#!/usr/bin/python
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse


prototxt = 'test.prototxt'
caffemodel = 'tattc_voc.caffemodel'

net = caffe.Net(prototxt, caffemodel, caffe.TEST)
#net = caffe.Net('/home/cgangee/code/TattDL/models/tattc_voc/faster_rcnn_end2end/test.prototxt', '/home/cgangee/code/TattDL/data/faster_rcnn_models/tattc_voc.caffemodel', caffe.TEST)
