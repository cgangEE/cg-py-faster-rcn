import tools._init_paths
from datasets.factory import get_imdb
from fast_rcnn.train import get_training_roidb
from datasets.compcars import compcars
from fast_rcnn.config import cfg
import pprint
import numpy as np
import cv2
import os
from utils.blob import im_list_to_blob
import Image, ImageDraw

def _init_caffe(cfg):
    """Initialize pycaffe in a training process.
    """

    import caffe
    # fix the random seeds (numpy and caffe) for reproducibility
    np.random.seed(cfg.RNG_SEED)
    caffe.set_random_seed(cfg.RNG_SEED)
    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(0)

def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
        im (ndarray): a color image in BGR order
    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []

    assert len(cfg.TEST.SCALES) == 1
    target_size = cfg.TEST.SCALES[-1]

    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
        im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    im_info = np.hstack((im.shape[:2], im_scale))[np.newaxis, :]
    processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_info


def im_proposals(net, im):
    """Generate RPN proposals on a single image."""
    blobs = {}
    blobs['data'], blobs['im_info'] = _get_image_blob(im)
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    blobs_out = net.forward(
            data=blobs['data'].astype(np.float32, copy=False),
            im_info=blobs['im_info'].astype(np.float32, copy=False))

    scale = blobs['im_info'][0, 2]

    boxes = blobs_out['rois'][:, 1:].copy() 
    scores = blobs_out['scores'].copy()
    return boxes, scores


cfg.TEST.RPN_PRE_NMS_TOP_N = -1     # no pre NMS filtering
cfg.TEST.RPN_POST_NMS_TOP_N = 2000  # limit top boxes after NMS


import caffe
_init_caffe(cfg)


imdb = get_imdb('compcars_trainval')

rpn_test_prototxt = os.path.join(
        cfg.MODELS_DIR, 'ZF', 'faster_rcnn_alt_opt', 'rpn_test.pt')
rpn_model_path = './output/faster_rcnn_alt_opt/compcars_trainval/zf_rpn_stage1_iter_80000.caffemodel'


# Load RPN and configure output directory
rpn_net = caffe.Net(rpn_test_prototxt, rpn_model_path, caffe.TEST)

nums = 10
boxes = [[] for _ in xrange(nums)]

for i in xrange(nums):
    im = cv2.imread(imdb.image_path_at(i))
    boxes[i], scores = im_proposals(rpn_net, im)
    im = Image.open(imdb.image_path_at(i))
    draw = ImageDraw.Draw(im)


    j=0
    for bbox in boxes[i]:
        if scores[j] > 0.5:
            draw.rectangle(tuple(bbox), outline= "green")
        j = j + 1

    im.show()

    print boxes[i][0]


#rpn_proposals = imdb_proposals(rpn_net, imdb)

