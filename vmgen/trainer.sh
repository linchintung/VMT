export PYTHONPATH=../lib:$PYTHONPATH

BASE_DIR=../Data/VMGen
DATA_DIR=$BASE_DIR/t2t_data
HPARAMS_SET=video2music_transformer
MODEL=mytransformer
PROBLEM=video_music_gen
TRAIN_DIR=$BASE_DIR/t2t_train/$PROBLEM

# rm -r $TRAIN_DIR

mkdir -p $DATA_DIR $TRAIN_DIR

HPARAMS=\
"label_smoothing=0.0,"\
"max_length=0,"\
"max_target_seq_length=768,"\
"hidden_size=512,"\
"batch_size=4"
# "batch_shuffle_size=1,"\

./mytrainer.py \
    --data_dir="${DATA_DIR}" \
    --hparams=${HPARAMS} \
    --hparams_set=${HPARAMS_SET} \
    --model=${MODEL} \
    --output_dir=${TRAIN_DIR} \
    --problem=${PROBLEM} \
    --train_steps=500000
    # --keep_checkpoint_max=10 \
    # --eval_early_stopping_steps=10


# batch * 1 * height * width * channel * #target

# batch * height * width
