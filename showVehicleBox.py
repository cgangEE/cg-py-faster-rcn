#!/usr/bin/python
import tools._init_paths
import cPickle
import os
from fast_rcnn.config import cfg
import numpy as np
from datasets.factory import get_imdb
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
import random


def _strToList(s):
    ret = []
    tmp = 0
    length = len(s)
    for i in range(length):
        if str.isdigit(s[i]):
            tmp = tmp * 10 + int(s[i])
        else:
            ret.append(tmp)
            tmp = 0
    return ret


def showImage(im, f):
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    for line in f:
        bbox = _strToList(line)
        ax.add_patch(
                    plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
                )

def showVehicleBox(image_set):

    imdb = get_imdb(image_set)
    num_images = len(imdb.image_index)

    for i in xrange(num_images):
        if random.randint(1, 100000) > 3:
            continue
        print i

        im_file = imdb.image_path_at(i)
        im = cv2.imread(im_file)
        if im is None:
            continue
        f = open(imdb.label_path_at(i), 'r')
        showImage(im, f)
        f.close()

    plt.show()

if __name__ == '__main__':
    showVehicleBox('vehicle_val')

