import os
from datasets.imdb import imdb
from fast_rcnn.config import cfg
import numpy as np
import scipy.sparse
import scipy.io as sio
import cPickle

class compcars(imdb):
    def __init__(self, image_set):
        imdb.__init__(self, 'compcars_' + image_set)
        self._image_set = image_set
        self._data_path = os.path.join(cfg.DATA_DIR, 'compcars')
        self._classes = ('__background__', 'car')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))

        self._image_index = self._load_image_set_index()
        self._image_ext = '.jpg'

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


    def gt_roidb(self):
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_compcars_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb


    def _strToList(self, s):
        ret = []
        tmp = 0
        length = len(s)
        for i in range(length):
            if str.isdigit(s[i]):
                tmp = tmp * 10 + int(s[i])
            else:
                ret.append(tmp-1)
                tmp = 0
        return ret


    def _load_compcars_annotation(self, index):
        filename = os.path.join(self._data_path, 'label', index + '.txt')
        label = open(filename)
        lineIdx = 0
        for line in label:
            lineIdx = lineIdx + 1
            if lineIdx == 3:
                ret = self._strToList(line)
                x1 = float(ret[0]) - 1
                y1 = float(ret[1]) - 1
                x2 = float(ret[2]) - 1
                y2 = float(ret[3]) - 1

        num_objs = 1

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        ix = 0
        cls = 1


        boxes[ix, :] = [x1, y1, x2, y2]
        gt_classes[ix] = cls
        overlaps[ix, cls] = 1.0
        seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}


    def image_path_at(self, i):
        return self.image_path_from_index(self._image_index[i])


    def image_path_from_index(self, index):
        image_path = os.path.join(self._data_path, 'image',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def rpn_roidb(self):
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
