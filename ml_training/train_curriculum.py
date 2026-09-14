#!/usr/bin/env python3
"""Curriculum training launcher -> agents.district.train_curriculum"""

import sys
import subprocess

if __name__ == '__main__':
    cmd = [sys.executable, '-m', 'agents.district.train_curriculum'] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))
