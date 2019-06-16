#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from video2music.logging import *
import video2music

# import coloredlogs, logging
# coloredlogs.install(level='DEBUG')
# logger = logging.getLogger()

import tensorflow as tf
# tf.enable_eager_execution()
from tensorflow import flags

from tensor2tensor.bin import t2t_trainer

# from video2music.utils import *

FLAGS = flags.FLAGS

def main(argv):
  demonstrate_logging()
  logger().debug(FLAGS.flag_values_dict())
  t2t_trainer.main(argv)


def console_entry_point():
  # tf.enable_eager_execution()
  # tf.logging.fatal("in mytrainer.py #30...")
  # import IPython
  # IPython.embed()
  # tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
