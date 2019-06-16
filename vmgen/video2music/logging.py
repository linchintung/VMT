#!/usr/bin/env python3
# -*- coding:utf-8 -*-

__author__    = 'Mu Yang <emfomy@gmail.com>'
__copyright__ = 'Copyright 2019'

import coloredlogs as _coloredlogs
import inspect as _inspect
import logging as _logging
import os as _os
import traceback as _traceback
import verboselogs as _verboselogs

# _fmt = '%(asctime)s %(name)32.32s:%(lineno)-4d %(levelname)8s %(message)s'
_fmt = '%(asctime)s %(pathname)64.64s:%(lineno)-4d %(levelname)8s %(message)s'
_field_styles = {
    'asctime': {'color': 'green', 'faint': True},
    'pathname': {'color': 'cyan', 'faint': True},
    'levelname': {'color': 'black', 'bold': True},
}
_level_styles = {
    'spam': {'color': 'magenta', 'faint': True},
    'debug': {'color': 'blue'},
    'verbose': {'color': 'magenta'},
    'info': {'color': 'cyan'},
    'notice': {'color': 'cyan', 'bold': True},
    # 'warning': {'color': 'yellow', 'bold': True},
    'warning': {'color': 'yellow'},
    'success': {'color': 'green', 'bold': True},
    'error': {'color': 'red', 'bold': True},
    'critical': {'background': 'red', 'bold': True},
}

_verboselogs.install()
_coloredlogs.install(level=5, fmt=_fmt, level_styles=_level_styles, field_styles=_field_styles)

from tensorflow import get_logger as _tf_get_logger
_tf_get_logger().handlers.clear()

_mpl_logger = _logging.getLogger('matplotlib')
_mpl_logger.setLevel(_logging.WARNING)

def logger():
    frm = _inspect.stack()[1]
    mod = _inspect.getmodule(frm[0])
    return _logging.getLogger(mod.__name__)

def exceptstr(e):
    return f'{e.__class__.__name__}: {e}'

def trace_error(e):
    msg = exceptstr(e)
    logger().error(msg)
    logger().warning('\n'+_traceback.format_exc())
    return msg

def demonstrate_logging():
    for name in _level_styles.keys():
        level = _coloredlogs.level_to_number(name)
        # if level < 50:
        logger().log(level, f'message with level {name} ({level})')
