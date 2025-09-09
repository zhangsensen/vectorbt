#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多时区因子雷达系统 - 模块化重构版

A high-performance quantitative trading system for multi-timezone factor analysis.
"""

__version__ = "2.0.0"
__author__ = "Multi-Timezone Radar Team"

from .core.data_structures import (
    MarketState,
    SignalStrength,
    FactorSignal,
    TimeFrame,
    TradingSession
)

from .core.config import Config
from .core.logger import setup_logger

__all__ = [
    "MarketState",
    "SignalStrength", 
    "FactorSignal",
    "TimeFrame",
    "TradingSession",
    "Config",
    "setup_logger"
]