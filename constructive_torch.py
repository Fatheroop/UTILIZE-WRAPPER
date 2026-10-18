"""
Constructive PyTorch Wrapper - Root Forwarder
============================================
Universal high-performance stage-wise progressive constructive learning framework.
Forwards to `core_models.constructive_torch` for backward compatibility.
The primary implementation is maintained in `core_models/constructive_torch.py`.
"""
import sys
import os

_repo_root = os.path.dirname(os.path.abspath(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from core_models.constructive_torch import *
from core_models.constructive_torch import __all__
