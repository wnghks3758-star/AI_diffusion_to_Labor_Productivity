"""Continuous DiD: configure, run, inspect."""

from .config import Config
from .pipeline import run

__all__ = ["Config", "run"]
