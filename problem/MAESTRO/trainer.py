#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import coloredlogs, logging, verboselogs
verboselogs.install()
coloredlogs.install(level=0)
logger = logging.getLogger(__name__)

from tensorflow import flags
import tensorflow as tf
tf.enable_eager_execution()

from magenta.models.score2perf import score2perf
from tensor2tensor.bin import t2t_trainer

FLAGS = flags.FLAGS

def main(argv):
  logger.debug(FLAGS.flag_values_dict())
  t2t_trainer.main(argv)


def console_entry_point():
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()