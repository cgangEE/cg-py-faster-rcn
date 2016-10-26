LOG="experiments/logs/log"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"
time ./tools/test_net.py --gpu 0 \
  --def models/pascal_voc/ZF/faster_rcnn_alt_opt/faster_rcnn_test.pt \
  --net output/faster_rcnn_alt_opt/tattoo_train/ZF_faster_rcnn_final.caffemodel \
  --imdb tattoo_val \
  --cfg experiments/cfgs/faster_rcnn_alt_opt.yml \

