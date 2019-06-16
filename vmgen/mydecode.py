#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from video2music.logging import *
import video2music

import tensorflow as tf
from tensorflow import flags

from tensor2tensor.bin import t2t_decoder

FLAGS = flags.FLAGS

def main(argv):
  demonstrate_logging()
  logger().debug(FLAGS.flag_values_dict())
  t2t_decoder.main(argv)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()