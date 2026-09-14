"""
Pytest configuration for test suite.
Sets up proper paths and environment for tests.
"""

import os
import sys
from pathlib import Path

# Get project root (ml_training/tests -> ml_training -> project_root)
tests_dir = Path(__file__).parent
ml_training_dir = tests_dir.parent
project_root = ml_training_dir.parent

# Change working directory to project root so game_data can be found
os.chdir(project_root)

# Add ml_training to Python path
sys.path.insert(0, str(ml_training_dir))
