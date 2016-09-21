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


def showImage(im, boxes):
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    for j in xrange( boxes.shape[0] ):
        bbox = boxes[j]
        if bbox[-1] >= 0.993822:
            ax.add_patch(
                    plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
                )
            ax.text(bbox[0], bbox[1] - 2,
                    '{:.3f}'.format(bbox[-1]),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')


def saveBoundingBox(imdb, i, boxes):
    f = open(imdb.label_path_at(i), 'w')
    for j in xrange( boxes.shape[0] ):
        bbox = boxes[j]
        if bbox[-1] >= 0.993822:
            b = bbox.astype(int)
            f.write('%d %d %d %d\n'% (b[0], b[1], b[2], b[3]))
    f.close()


def generateVehicleBox(image_set):
    cache_file = os.path.join(cfg.ROOT_DIR,
    'output/faster_rcnn_alt_opt/'+image_set+'/ZF_faster_rcnn_final/detections.pkl')

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            boxes = cPickle.load(fid)


    imdb = get_imdb(image_set)
    num_images = len(imdb.image_index)

    print num_images
    for i in xrange(num_images):
        print i
        im_file = imdb.image_path_at(i)
        im = cv2.imread(im_file)
        if im is None: 
            continue

        saveBoundingBox(imdb, i, boxes[1][i])
        #showImage(im, boxes[1][i])


    plt.show()


if __name__ == '__main__':
    generateVehicleBox('vehicle_val')
    
