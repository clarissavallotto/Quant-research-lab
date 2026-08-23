"""
Quantitative research demonstration package.
"""

from .experiment import run, load_config
from .presentation import generate_research_report

__all__ = [
    "run",
    "load_config",
    "generate_research_report",
]
