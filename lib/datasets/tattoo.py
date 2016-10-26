import os
from datasets.imdb import imdb
from fast_rcnn.config import cfg
from datasets.getBoundingBox import getBoundingBox

import numpy as np
import scipy.sparse
import scipy.io as sio
import cPickle
import uuid
import PIL

class tattoo(imdb):
    def __init__(self, image_set):
        imdb.__init__(self, 'tattoo_' + image_set)
        self._image_set = image_set
        self._data_path = os.path.join(cfg.DATA_DIR, 'tattoo')

        self._classes = ('background', 'tattoo')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))

        self._image_index = self._load_image_set_index()
        self._image_ext = '.jpg'
        self._roidb_handler = self.gt_roidb


        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp8'

        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)


    def _load_image_set_index(self):
        image_set_file = os.path.join(self._data_path, 
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def update(self, x, w):
        x = max(0, x)
        x = min(x, w-1)
        return x

    def gt_roidb(self):
        gt_roidb = []

        boxes = getBoundingBox(os.path.join(self._data_path, 
                    'tattoo_annotations.txt'))
        
        idx = 0
        image_index = self._image_index

        for line, bb in boxes:
            line = line[:-1]

            if (idx == len(image_index)):
                break
            if image_index[idx].find(line) != -1:

                w, h = PIL.Image.open(self.image_path_at(idx)).size

                idx += 1
                num_objs = len(bb) 

                bxes = np.zeros((num_objs, 4), dtype=np.uint16)
                gt_classes = np.zeros((num_objs), dtype=np.int32)
                overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
                seg_areas = np.zeros((num_objs), dtype=np.float32)

                cls = 1
                for ix, box in enumerate(bb):
                    x1, y1, x2, y2 = box
                    x1 = self.update(x1, w)
                    x2 = self.update(x2, w)
                    y1 = self.update(y1, h)
                    y2 = self.update(y2, h)

                    bxes[ix, :] = [x1, y1, x2, y2]
                    gt_classes[ix] = cls
                    overlaps[ix, cls] = 1.0
                    seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

                
                overlaps = scipy.sparse.csr_matrix(overlaps)

                gt_roidb.append({'boxes' : bxes,
                    'gt_classes': gt_classes,
                    'gt_overlaps' : overlaps,
                    'flipped' : False,
                    'seg_areas' : seg_areas})

        return gt_roidb


    def image_path_at(self, i):
        return self.image_path_from_index(self._image_index[i])


    def image_path_from_index(self, index):
        image_path = os.path.join(self._data_path, 
                                  index)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def rpn_roidb(self):
        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb


    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True
