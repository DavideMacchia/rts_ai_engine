#!/usr/bin/env python
"""
Test runner for ML training tests.
Ensures proper environment setup before running tests.
"""

import sys
import os
from pathlib import Path

# Get paths
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent
tests_dir = script_dir / 'tests'

# Change to project root so game_data can be found
os.chdir(project_root)

# Add rl to path
sys.path.insert(0, str(script_dir))

# Now run pytest
import pytest

if __name__ == '__main__':
    # Convert relative paths to absolute
    args = []
    for arg in sys.argv[1:]:
        if arg.startswith('tests/'):
            # Convert relative test path to absolute
            args.append(str(script_dir / arg))
        else:
            args.append(arg)

    # Default to running all tests
    if not args or not any(str(script_dir) in str(a) or a.startswith('-') for a in args):
        args = [str(tests_dir), '-v']

    exit_code = pytest.main(args)
    sys.exit(exit_code)
