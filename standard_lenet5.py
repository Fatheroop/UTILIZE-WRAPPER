"""
Standard LeNet-5 - Root Forwarder
=================================
End-to-end standard LeNet-5 baseline.
Forwards to `core_models.standard_lenet5` for backward compatibility.
The primary implementation is maintained in `core_models/standard_lenet5.py`.
"""
import sys
import os

_repo_root = os.path.dirname(os.path.abspath(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from core_models.standard_lenet5 import *
