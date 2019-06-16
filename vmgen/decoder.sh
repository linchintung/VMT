export PYTHONPATH=../lib:$PYTHONPATH

BASE_DIR=../Data/VMGen
DATA_DIR=$BASE_DIR/t2t_data
HPARAMS_SET=video2music_transformer
MODEL=mytransformer
PROBLEM=video_music_gen
DECODE_DIR=$BASE_DIR/t2t_train/chosed_model
DECODE_FILE=$DATA_DIR/${PROBLEM}-test.tfrecord-00000-of-00001

rm -r $DECODE_DIR/decode*

DECODE_HPARAMS=\
"alpha=0,"\
"beam_size=1,"\
"extra_length=1024"

HPARAMS=\
"sampling_method=random,"\
"max_target_seq_length=1024,"\
"hidden_size=512,"\
"batch_size=4"

./mydecode.py \
  --data_dir="${DATA_DIR}" \
  --decode_hparams="${DECODE_HPARAMS}" \
  --hparams=${HPARAMS} \
  --hparams_set=${HPARAMS_SET} \
  --model=${MODEL} \
  --problem=${PROBLEM} \
  --output_dir=${DECODE_DIR} \
  --decode_from_dataset

#   --decode_interactive \