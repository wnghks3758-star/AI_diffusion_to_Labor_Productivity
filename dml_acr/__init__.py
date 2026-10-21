"""Configurable, memory bounded continuous-dose DiD exploration."""
from .config import AnalysisConfig

__version__ = '0.1.0'

def run_analysis(config, output_dir=None):
    from .pipeline import run_analysis as run
    return run(config, output_dir)
