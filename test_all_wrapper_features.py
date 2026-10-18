"""
Wrapper Features Audit - Root Runner
====================================
Runs the end-to-end wrapper feature audit from `tests/test_all_wrapper_features.py`.
"""
import sys
import os

_repo_root = os.path.dirname(os.path.abspath(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from tests.test_all_wrapper_features import *

if __name__ == "__main__":
    run_all_feature_verifications()
