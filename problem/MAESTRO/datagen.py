#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import coloredlogs, logging
coloredlogs.install(level='DEBUG')
logger = logging.getLogger()

from tensorflow import flags
import tensorflow as tf

from magenta.models.score2perf import score2perf
from tensor2tensor.bin import t2t_datagen


FLAGS = flags.FLAGS

def main(argv):
  logger.debug(FLAGS.flag_values_dict())
  t2t_datagen.main(argv)


def console_entry_point():
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()