"""Vectorized portfolio analyzer package."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = ["FinalWorkingVectorBT"]

if TYPE_CHECKING:  # pragma: no cover
    from .core.main_analyzer import FinalWorkingVectorBT as _FinalWorkingVectorBT


def __getattr__(name: str) -> Any:
    if name == "FinalWorkingVectorBT":
        module = import_module("longport.vectorized.portfolio_analyzer.core.main_analyzer")
        return module.FinalWorkingVectorBT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
