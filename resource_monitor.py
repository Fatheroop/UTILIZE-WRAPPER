"""
Resource Monitor - Root Forwarder
=================================
Live CPU, RSS RAM, and training telemetry tracking.
Forwards to `utils.resource_monitor` for backward compatibility.
The primary implementation is maintained in `utils/resource_monitor.py`.
"""
import sys
import os

_repo_root = os.path.dirname(os.path.abspath(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from utils.resource_monitor import *
