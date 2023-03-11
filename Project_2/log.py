import os
import time
import logging
import datetime
from logging import handlers


def record(**kwargs):
    filename = kwargs['filename']
    level = kwargs['level']
    datefmt = kwargs.pop('datefmt', None)
    format = kwargs.pop('format', None)
    if level is None:
        level = logging.INFO
    if datefmt is None:
        datefmt = '%Y-%m-%d %H:%M:%S'
    if format is None:
        format = '%(asctime)s [%(module)s] %(levelname)s [%(lineno)d] %(message)s'

    if not os.path.exists(filename):
        with open(filename, 'w') as f:  # 创建文件
            print("新建日志文件",f)
    log = logging.getLogger(filename)
    format_str = logging.Formatter(format, datefmt)
    th = handlers.TimedRotatingFileHandler(filename=filename, backupCount=7, when='midnight', encoding='utf-8')
    th.setFormatter(format_str)
    th.setLevel(level)
    log.addHandler(th)
    log.setLevel(level)
    return log
