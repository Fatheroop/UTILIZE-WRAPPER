"""
Core Model - Root Forwarder
===========================
Constructive layer-wise neural network for vector and tabular tasks.
Forwards to `core_models.core_model` for backward compatibility.
The primary implementation is maintained in `core_models/core_model.py`.
"""
import sys
import os

_repo_root = os.path.dirname(os.path.abspath(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from core_models.core_model import *
