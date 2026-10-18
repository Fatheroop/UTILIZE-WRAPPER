"""
Test Suite - Root Runner
========================
Runs the complete 21-test automated verification suite from `tests/test_suite.py`.
"""
import sys
import os
import unittest

_repo_root = os.path.dirname(os.path.abspath(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from tests.test_suite import *

if __name__ == "__main__":
    unittest.main()
