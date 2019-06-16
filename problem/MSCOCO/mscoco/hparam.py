"""Hyperparameters."""
from tensor2tensor.utils import registry
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import modalities
from tensor2tensor.layers import common_image_attention as cia

@registry.register_hparams
def hparam_img2txt_base():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.hidden_size = 512
  hparams.batch_size = 4
  hparams.max_length = 3075
  hparams.dropout = 0.0
  hparams.clip_grad_norm = 0.  # i.e. no gradient clipping
  hparams.optimizer_adam_epsilon = 1e-9
  hparams.learning_rate_decay_scheme = "noam"
  hparams.learning_rate = 0.1
  hparams.learning_rate_warmup_steps = 4000
  hparams.initializer_gain = 0.2
  hparams.num_hidden_layers = 6
  hparams.initializer = "uniform_unit_scaling"
  hparams.weight_decay = 0.0
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.98
  hparams.label_smoothing = 0.0
  # hparams.bottom["targets"] = modalities.image_channel_embeddings_bottom
  # hparams.top["targets"] = modalities.identity_top
  hparams.norm_type = "layer"
  hparams.layer_prepostprocess_dropout = 0.0
  hparams.add_hparam("filter_size", 512)  # Add new ones like this.

  # attention-related flags
  hparams.add_hparam("num_heads", 8)
  hparams.add_hparam("attention_key_channels", 0)
  hparams.add_hparam("attention_value_channels", 0)
  hparams.add_hparam("ffn_layer", "conv_hidden_relu")
  # All hyperparameters ending in "dropout" are automatically set to 0.0
  # when not in training mode.
  hparams.add_hparam("attention_dropout", 0.0)
  hparams.add_hparam("relu_dropout", 0.0)
  hparams.add_hparam("pos", "timing")  # timing, none
  hparams.add_hparam("nbr_decoder_problems", 1)
  hparams.add_hparam("num_output_layers", 3)
  hparams.add_hparam("block_size", 1)

  # dilated attention based flags
  hparams.add_hparam("gap_sizes", [2, 4, 8, 16, 32, 64, 2, 4, 8, 16, 32, 64])

  # image size related flags
  # assuming that the image has same height and width
  hparams.add_hparam("img_len", 32)
  hparams.add_hparam("num_channels", 3)
  # Local attention params
  hparams.add_hparam("local_and_global_att", False)
  hparams.add_hparam("block_length", 256)
  hparams.add_hparam("block_width", 128)
  hparams.add_hparam("num_encoder_layers", 4)
  hparams.add_hparam("num_decoder_layers", 12)
  hparams.add_hparam("dec_attention_type", cia.AttentionType.LOCAL_1D)
  hparams.add_hparam("block_raster_scan", False)

  # multipos attention params
  hparams.add_hparam("q_filter_width", 1)
  hparams.add_hparam("kv_filter_width", 1)

  hparams.add_hparam("likelihood", cia.DistributionType.CAT)
  hparams.add_hparam("unconditional", False)  # unconditional generation

  # parameters of discretized mixture of logistics loss from pixel cnn++
  hparams.add_hparam("num_mixtures", 10)

  # These parameters are only used when ffn_layer=="local_moe_tpu"
  hparams.add_hparam("moe_overhead_train", 1.0)
  hparams.add_hparam("moe_overhead_eval", 2.0)
  hparams.moe_num_experts = 8
  hparams.moe_loss_coef = 1e-3

  # These parameters are for relative attention
  hparams.add_hparam("shared_rel", False)  # share relative embeddings
  return hparams

  # """Hparams for ImageCaptioning"""
  # hparams = common_hparams.basic_params1()
  # hparams.daisy_chain_variables = False
  # hparams.initializer = "uniform_unit_scaling"
  # hparams.initializer_gain = 1.0
  # hparams.weight_decay = 0.0
  # hparams.dropout = 0.0
  # hparams.layer_prepostprocess_dropout = 0.0

  # # image size related flags
  # # please provide them in here accordingly.

  # # base setting for the attention module
  # hparams.batch_size = 32
  # hparams.use_fixed_batch_size = True

  # hparams.hidden_size = 128
  # # hparams.add_hparam("attention_mechanism", "bahdanau")
  # # hparams.max_input_seq_length = -1
  # # hparams.max_target_seq_length = 30
  # hparams.add_hparam("likelihood", cia.DistributionType.DMOL)

  # return hparams