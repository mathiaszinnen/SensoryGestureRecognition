CONFIG=$1
RUN=$2

CKPT=/home/woody/iwi5/iwi5093h/sniffyart/workdirs/combined/gesture_detect/swinb_$RUN/best
OUT_FILE=/home/woody/iwi5/iwi5093h/sniffyart/workdirs/combined/gesture_detect/swinb_$RUN/preds

python test.py $CONFIG $CKPT --cfg-options test_evaluator.format_only=True test_evaluator.outfile_prefix=$OUT_FILE