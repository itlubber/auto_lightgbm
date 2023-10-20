# -*- coding: utf-8 -*-
"""
@Time    : 2023/05/21 16:23
@Author  : itlubber
@Site    : itlubber.art
"""

from .utils.logger import logger
from .model import auto_lightgbm, auto_logistic

__version__ = "0.1.3"
__all__ = ("__version__", "auto_lightgbm", "auto_logistic", "logger")
