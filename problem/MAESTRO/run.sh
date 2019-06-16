export PYTHONPATH=../../lib:$PYTHONPATH

# PROBLEM=score2perf_maestro_language_uncropped_aug
PROBLEM=score2perf_maestro_absmel2perf_5s_to_30s_aug10x
MODEL=transformer
HPARAMS_SET=score2perf_transformer_base

BASE_DIR=../../Data
WS=MAESTRO
DATA_DIR=$BASE_DIR/$WS/t2t_data
TMP_DIR=$BASE_DIR/$WS/tmp/t2t_datagen
TRAIN_DIR=$BASE_DIR/$WS/t2t_train/$PROBLEM/$MODEL-$HPARAMS_SET

rm -r $TRAIN_DIR

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

# ./datagen.py \
#   --data_dir="${DATA_DIR}" \
#   --problem=${PROBLEM} \
#   --alsologtostderr

HPARAMS=\
"label_smoothing=0.0,"\
"max_length=0,"\
"max_target_seq_length=2048"

# t2t_trainer \
./trainer.py \
  --data_dir="${DATA_DIR}" \
  --hparams=${HPARAMS} \
  --hparams_set=${HPARAMS_SET} \
  --model=${MODEL} \
  --output_dir=${TRAIN_DIR} \
  --problem=${PROBLEM} \
  --train_steps=1000