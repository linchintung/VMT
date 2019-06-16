export PYTHONPATH=../../lib:$PYTHONPATH

PROBLEM=image_ms_coco_characters
MODEL=imagetransformer
HPARAMS_SET=hparam_img2txt_base

BASE_DIR=../../Data
WS=MSCOCO
DATA_DIR=$BASE_DIR/$WS/t2t_data
TMP_DIR=$BASE_DIR/$WS/tmp/t2t_datagen
TRAIN_DIR=$BASE_DIR/$WS/t2t_train/$PROBLEM/$MODEL-$HPARAMS_SET

rm -r $TRAIN_DIR

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

# Generate data
#../../lib/tensor2tensor/bin/t2t-datagen \
#  --data_dir=$DATA_DIR \
#  --tmp_dir=$TMP_DIR \
#  --problem=$PROBLEM

# Train
# *  If you run out of memory, add --hparams='batch_size=1024'.
# ../../lib/tensor2tensor/bin/t2t-trainer \
./trainer.py \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS_SET \
  --output_dir=$TRAIN_DIR
