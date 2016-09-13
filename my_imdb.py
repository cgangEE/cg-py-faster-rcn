import tools._init_paths
from datasets.factory import get_imdb
from fast_rcnn.train import get_training_roidb
from datasets.compcars import compcars

class f(object):
    def __init__(self):
        x = self.gan

    def gan(self):
        print 'ft'

imdb = get_imdb('compcars_trainval')
imdb.set_proposal_method('gt')

roidb = get_training_roidb(imdb)

