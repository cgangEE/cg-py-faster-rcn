import tools._init_paths
from datasets.factory import get_imdb
from fast_rcnn.train import get_training_roidb
from datasets.compcars import compcars
from datasets.vehicle import vehicle


imdb = get_imdb('vehicle_val')
#imdb.set_proposal_method('gt')

#roidb = get_training_roidb(imdb)

