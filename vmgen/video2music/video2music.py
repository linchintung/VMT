from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
import numpy as np

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
# from tensor2tensor.data_generators import video_utils
from video2music import myvideo_utils as video_utils
from tensor2tensor.layers import modalities as t2t_modalities
from tensor2tensor.utils import registry
# from tensor2tensor.models import image_transformer_2d
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_image_attention as cia
# from tensor2tensor.models import transformer
from video2music import mytransformer as transformer


import tensorflow as tf

# from video2music.utils import *

from video2music import music_encoders

NUM_VELOCITY_BINS = 32
STEPS_PER_SECOND = 100
MIN_PITCH = 21
MAX_PITCH = 108

@registry.register_hparams
def video2music_transformer():
  """Set of hyperparameters."""
  # hparams = common_hparams.basic_params1()
  hparams = transformer.mytransformer_base()
  hparams.video_num_input_frames = 30
  hparams.video_num_target_frames = 0
  hparams.bottom["inputs"] = t2t_modalities.video_bottom
  hparams.add_hparam("shuffle_buffer_size", 128)
  # hparams.add_hparam("shuffle_buffer_size", 128)

  # """Base params for img2img 2d attention."""
  # # learning related flags
  # hparams.layer_preprocess_sequence = "n"
  # hparams.layer_postprocess_sequence = "da"
  # """enc_attention_type choices:
  # LOCAL_1D = "local_1d"
  # LOCAL_2D = "local_2d"
  # GLOBAL = "global"
  # GLOCAL = "global_local"
  # DILATED = "dilated"
  # MOE_LOCAL_1D = "moe_local1d"
  # LOCAL_BLOCK = "local_block"
  # NON_CAUSAL_1D = "local_1d_noncausal"
  # RELATIVE_LOCAL_1D = "rel_local_1d"
  # """
 
  # hparams.learning_rate = 0.2
  # hparams.layer_prepostprocess_dropout = 0.1
  # hparams.learning_rate_warmup_steps = 12000
  # hparams.filter_size = 2048
  # hparams.num_encoder_layers = 4
  # # hparams.add_hparam("enc_attention_type", cia.AttentionType.LOCAL_2D)
  # # hparams.bottom["inputs"] = t2t_modalities.video_bitwise_bottom
  # hparams.add_hparam("enc_attention_type", cia.AttentionType.GLOBAL)
  # hparams.bottom["inputs"] = t2t_modalities.video_bottom
  # # hparams.bottom["inputs"] = t2t_modalities.image_channel_compress_bottom
  # hparams.block_raster_scan = True

  """Base params for image_transformer2d_base  attention."""
  # hparams.filter_size = 512

  # # attention-related flags
  # hparams.ffn_layer = "conv_hidden_relu"
  # # All hyperparameters ending in "dropout" are automatically set to 0.0
  # # when not in training mode.

  # hparams.add_hparam("num_output_layers", 3)
  # hparams.add_hparam("block_size", 1)

  # # image size related flags
  # # assuming that the image has same height and width
  # hparams.add_hparam("img_len", 32)
  # hparams.add_hparam("num_channels", 3)
  # # Local attention params
  # hparams.add_hparam("local_and_global_att", False)
  # hparams.add_hparam("block_length", 256)
  # hparams.add_hparam("block_width", 128)
  # # Local 2D attention params
  # hparams.add_hparam("query_shape", (16, 16))
  # hparams.add_hparam("memory_flange", (16, 32))
  # hparams.num_decoder_layers = 8
  # # attention type related params
  # hparams.add_hparam("dec_attention_type", cia.AttentionType.LOCAL_2D)

  # # multipos attention params
  # hparams.add_hparam("q_filter_width", 1)
  # hparams.add_hparam("kv_filter_width", 1)

  # hparams.add_hparam("unconditional", False)  # unconditional generation

  # # relative embedding hparams
  # hparams.add_hparam("shared_rel", False)




  return hparams

@registry.register_problem('video_music_gen')
class VideoMusicGen(video_utils.VideoProblem):
# class VideoMusicGen(problem.Problem):
  """My BGM (background music) dataset."""

  @property
  def num_channels(self):
    return 3

  @property
  def frame_height(self):
    return 256

  @property
  def frame_width(self):
    return 256

  @property
  def is_generate_per_split(self):
    return True

  # num_train_files * num_videos * num_frames
  @property
  def total_number_of_frames(self):
    return 10 * 378 * 30

  def max_frames_per_video(self, hparams):
    return 40

  @property
  def random_skip(self):
    return False

  def eval_metrics(self):
    return []

  @property
  def only_keep_videos_from_0th_frame(self):
    return True

  @property
  def use_not_breaking_batching(self):
    return True

  @property
  def add_eos_symbol(self):
    return True

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each."""
    return [
        {"split": problem.DatasetSplit.TRAIN, "shards": 1},
        {"split": problem.DatasetSplit.EVAL, "shards": 1},
        {"split": problem.DatasetSplit.TEST, "shards": 1}]

  @property
  def extra_reading_spec(self):
    """Additional data fields to store on disk and their decoders."""
    # from myscore2perf.utils import print_frameinfo, print_subtitle
    # from inspect import currentframe, getframeinfo
    # print_frameinfo(getframeinfo(currentframe()))

    data_fields = {
        "frame_number": tf.FixedLenFeature([1], tf.int64),
    }
    decoders = {
        "frame_number": tf.contrib.slim.tfexample_decoder.Tensor(
            tensor_key="frame_number"),
    }
    return data_fields, decoders

  def example_reading_spec(self):
    extra_data_fields, extra_data_items_to_decoders = self.extra_reading_spec

    data_fields = {
        "image/encoded": tf.FixedLenFeature((), tf.string),
        "image/format": tf.FixedLenFeature((), tf.string),
    }
    data_fields.update(extra_data_fields)

    data_items_to_decoders = {
        "frame":
            tf.contrib.slim.tfexample_decoder.Image(
                image_key="image/encoded",
                format_key="image/format",
                shape=[self.frame_height, self.frame_width, self.num_channels],
                channels=self.num_channels),
    }
    data_items_to_decoders.update(extra_data_items_to_decoders)

    # label_key = "image/class/label"
    label_key = "targets"
    # data_fields[label_key] = tf.FixedLenFeature((1,), tf.int64)
    data_fields[label_key] = tf.VarLenFeature(tf.int64)
    data_items_to_decoders[
        "targets"] = tf.contrib.slim.tfexample_decoder.Tensor(label_key)
    # data_fields["targets"] = tf.VarLenFeature(tf.int64)
    # data_items_to_decoders = None

    return data_fields, data_items_to_decoders
    
  def hparams(self, defaults, unused_model_hparams):
    del unused_model_hparams
    perf_encoder = self.get_feature_encoders()["targets"]
    p = defaults
    p.modality = {"inputs": t2t_modalities.ModalityType.VIDEO,
                  "targets": t2t_modalities.ModalityType.SYMBOL,
    }
    p.vocab_size = {"inputs": 256,
                    "targets": perf_encoder.vocab_size
    }
    p.input_space_id = problem.SpaceID.IMAGE
    p.target_space_id = problem.SpaceID.IMAGE_LABEL
  
  def performance_encoder(self):
    """Encoder for target performances."""
    return music_encoders.MidiPerformanceEncoder(
        steps_per_second=STEPS_PER_SECOND,
        num_velocity_bins=NUM_VELOCITY_BINS,
        min_pitch=MIN_PITCH,
        max_pitch=MAX_PITCH,
        add_eos=self.add_eos_symbol)

  def feature_encoders(self, data_dir):
    del data_dir
    encoders = {
        "inputs": text_encoder.ImageEncoder(),
        'targets': self.performance_encoder()
    }
    return encoders

