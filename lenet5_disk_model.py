"""
Constructive LeNet-5 - Root Forwarder
=====================================
Chunked disk-streamed progressive LeNet-5 architecture.
Forwards to `core_models.lenet5_disk_model` for backward compatibility.
The primary implementation is maintained in `core_models/lenet5_disk_model.py`.
"""
import sys
import os

_repo_root = os.path.dirname(os.path.abspath(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from core_models.lenet5_disk_model import *
