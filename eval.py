import tools._init_paths
import cPickle
import os
from fast_rcnn.config import cfg
import numpy as np
from datasets.factory import get_imdb
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt

def frange(x, y, jump):
    ret = []
    while x < y:
        ret.append(x)
        x += jump
    return ret

def vis_detections(im, dets):
    """Draw detected bounding boxes."""

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    inds = dets.shape[0]

    for i in xrange(inds):
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )

    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def IoU(box, gt_box):
    b = np.zeros(4)
    b[0:2] = np.stack((box[0:2], gt_box[0:2])).max(0)
    b[2:4] = np.stack((box[2:4], gt_box[2:4])).min(0)

    iw = b[3] - b[1] + 1
    ih = b[2] - b[0] + 1
    if iw > 0 and ih > 0:
        ua = (box[3] - box[1] + 1) * (box[2] - box[0] + 1) + \
             (gt_box[3] - gt_box[1] + 1) * (gt_box[2] - gt_box[0] + 1) - \
             iw * ih
        return iw * ih / ua

    return 0.0

def eval():
    cache_file = os.path.join(cfg.ROOT_DIR,
    'output/faster_rcnn_alt_opt/carsample_val/ZF_faster_rcnn_final/detections.pkl')

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            boxes = cPickle.load(fid)

    imdb = get_imdb('carsample_val')
    num_images = len(imdb.image_index)
    roidb = imdb.roidb

    bb = []
    gt_box_num = 0

    for i in xrange(num_images):
        roidb[i]['detected'] = [False]*len(roidb[i]['boxes'])
        gt_box_num = gt_box_num + len(roidb[i]['boxes'])

        for j in xrange(1, imdb.num_classes):
            for k in xrange(boxes[j][i].shape[0]):
                bb.append(np.hstack( (i, boxes[j][i][k])))
               
    bb = sorted(bb, key=lambda x:x[-1], reverse=True)


    num_bb = len(bb)
    tp = np.zeros(num_bb)
    fp = np.zeros(num_bb)

    for bb_idx in xrange(num_bb):
        image_idx = int(bb[bb_idx][0])
        box = bb[bb_idx][1:-1]

        gt_boxes = roidb[image_idx]['boxes']

        iouMax = 0.0
        max_iou_box_idx = 0
        for box_idx in xrange(len(gt_boxes)):
            iou = IoU(box, gt_boxes[box_idx])
            if iou > iouMax:
                iouMax = iou
                max_iou_box_idx = box_idx
            
        if iouMax > 0.5:
            if not roidb[image_idx]['detected'][max_iou_box_idx]:
                tp[bb_idx] = 1
                roidb[image_idx]['detected'][max_iou_box_idx] = True
            else:
                fp[bb_idx] = 1
        else:
            fp[bb_idx] = 1
            

    tp = tp.cumsum()
    fp = fp.cumsum()
    rec = tp / gt_box_num;
    prec = tp / (fp + tp)

    ap = 0.0
    p = 0.0
    for x in range(11):
        tmp = prec[rec>=x/10.0]
        if tmp.shape == (0,):
            p = 0.0;
        else:
            p = tmp.max()
        ap += p / 11.0
        
    print 'AP is %.10lf' % ap

'''
    plt.plot(rec, prec, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()
'''

if __name__ == '__main__':
    eval()

