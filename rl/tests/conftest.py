"""
Pytest configuration for test suite.
Sets up proper paths and environment for tests.
"""

import os
import sys
from pathlib import Path

# Get project root (rl/tests -> rl -> project_root)
tests_dir = Path(__file__).parent
rl_dir = tests_dir.parent
project_root = rl_dir.parent

# Change working directory to project root so game_data can be found
os.chdir(project_root)

# Add rl to Python path
sys.path.insert(0, str(rl_dir))
