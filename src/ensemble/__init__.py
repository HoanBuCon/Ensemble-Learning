"""
Ensemble Module
===============

Independent ensemble learning package. Operates entirely on cached
``.npy`` probability files — zero PyTorch dependency.

Classes:
    - :class:`HardVoting`
    - :class:`SoftVoting`
    - :class:`WeightedVoting`
    - :class:`StackingEnsemble`
    - :class:`OOFGenerator`
"""

from src.ensemble.voting import HardVoting, SoftVoting, WeightedVoting
from src.ensemble.stacking import StackingEnsemble
from src.ensemble.oof import OOFGenerator

__all__ = [
    "HardVoting",
    "SoftVoting",
    "WeightedVoting",
    "StackingEnsemble",
    "OOFGenerator",
]
