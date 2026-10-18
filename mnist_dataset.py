"""
MNIST Dataset Loader - Root Forwarder
=====================================
Zero-overhead binary dataset loader.
Forwards to `utils.mnist_dataset` for backward compatibility.
The primary implementation is maintained in `utils/mnist_dataset.py`.
"""
import sys
import os

_repo_root = os.path.dirname(os.path.abspath(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from utils.mnist_dataset import *
