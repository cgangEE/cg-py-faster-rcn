LOG="./log"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

#./experiments/scripts/faster_rcnn_alt_opt.sh 0 ZF pascal_voc
./tools/train_faster_rcnn_alt_opt.py --gpu 0 --net_name ZF --weights data/imagenet_models/ZF.v2.caffemodel --imdb tattoo_train --cfg experiments/cfgs/faster_rcnn_alt_opt.yml
