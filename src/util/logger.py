import colorama
from logging import *
import logging
# import logging.LOGBOOK
import os
import sys


def configure_logging(file_name=None):
    log_fmt = "{}{}%(asctime)s{} %(message)s".format(colorama.Style.DIM, colorama.Fore.WHITE, colorama.Style.RESET_ALL)
    date_fmt = "%Y-%m-%d %H:%M:%S"

    addLoggingLevel("LOGBOOK", 1000)

    root_logger = logging.getLogger("")
    root_logger.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=log_fmt, datefmt=date_fmt)
    console.setFormatter(formatter)
    skip_logbook_filter = SkipLogbookFilter()
    console.addFilter(skip_logbook_filter)
    root_logger.addHandler(console)

    if file_name is not None:
        logbook = logging.FileHandler(filename=file_name, mode="a", encoding="utf-8")
        #         print("logger setup")
        logbook.setLevel(logging.INFO)
        fmt = "{}{}%(asctime)s{} %(message)s"
        logbook_formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)
        logbook.setFormatter(logbook_formatter)
        root_logger.addHandler(logbook)


# this piece is taken from https://github.com/visinf/self-mono-sf/blob/master/core/logger.py
def addLoggingLevel(level_name, level_num, method_name=None):
    if not method_name:
        method_name = level_name.lower()
    if hasattr(logging, level_name):
        raise AttributeError('{} already defined in logging module'.format(level_name))
    if hasattr(logging, method_name):
        raise AttributeError('{} already defined in logging module'.format(method_name))
    if hasattr(logging.getLoggerClass(), method_name):
        raise AttributeError('{} already defined in logger class'.format(method_name))

    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(level_num):
            self._log(level_num, message, args, **kwargs)

    def logToRoot(message, *args, **kwargs):
        logging.log(level_num, message, *args, **kwargs)

    logging.addLevelName(level_num, level_name)
    setattr(logging, level_name, level_num)
    setattr(logging.getLoggerClass(), method_name, logForLevel)
    setattr(logging, method_name, logToRoot)


class SkipLogbookFilter(logging.Filter):
    def filter(self, record):
        return record.levelno != logging.LOGBOOK


class LoggingBlock:
    def __init__(self, title, emph):
        self._emph = emph
        bright = colorama.Style.BRIGHT
        cyan = colorama.Fore.CYAN
        reset = colorama.Style.RESET_ALL
        if emph:
            #             print("log %s" % emph)
            logging.info("%s==>%s %s%s%s" % (cyan, reset, bright, title, reset))

    #         else:
    # #             print("log ++ %s" % emph)
    #             logging.info(title)
    def __enter__(self):
        #         sys.modules[__name__].global_indent += 2
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        #         sys.modules[__name__].global_indent -= 2
        return