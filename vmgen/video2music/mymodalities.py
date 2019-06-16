from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_image_attention as cia
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import common_video

import tensorflow as tf
import tensorflow_probability as tfp

def _video_channel_compress_bottom(inputs, model_hparams, name="bottom"):
  """Compresses channel-wise input pixels into whole pixel representions.

  Perform conversion of RGB pixel values to a real number in the range -1 to
  1. This combines pixel channels to form a representation of shape
  [img_len, img_len].

  Args:
    inputs: Tensor representing RGB pixel intensities as integers, of shape
      [batch, img_len, img_len, channels].
    model_hparams: HParams, model hyperparmeters.
    name: string, scope.

  Returns:
    body_input: Tensor of shape
      [batch, img_len, img_len, model_hparams.hidden_size].
  """
  num_channels = 3
  with tf.variable_scope(name):
    inputs = tf.to_float(inputs)
    hp = model_hparams
    if hp.mode != tf.estimator.ModeKeys.PREDICT:
      tf.summary.image(
          "inputs",
          common_layers.tpu_safe_image_summary(inputs),
          max_outputs=2)
    inputs = common_layers.convert_rgb_to_symmetric_real(inputs)

    # Reshape inputs to apply convolutions across [img_len, img_len*channels].
    inputs_shape = common_layers.shape_list(inputs)
    inputs = tf.reshape(
        inputs, [-1, inputs_shape[1], inputs_shape[2] * inputs_shape[3], 1])

    # Compress RGB intensities for each pixel using a convolution.
    outputs = tf.layers.conv2d(
        inputs,
        model_hparams.hidden_size,
        kernel_size=(1, num_channels),
        padding="VALID",
        strides=(1, num_channels),
        activation=tf.nn.relu,
        name="conv_input")
    return outputs


def video_channel_compress_bottom(x, model_hparams, vocab_size):
  del vocab_size  # unused arg
  return _video_channel_compress_bottom(x, model_hparams, "input_bottom")
