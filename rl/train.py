#!/usr/bin/env python3
"""
Training launcher.

The district-tier agent lives in `agents/district/`. Prefer running the modules
directly, which is what this launcher does:

    python -m agents.district.pretrain_bc      # BC + critic warm-up (the deliverable)
    python -m agents.district.train            # MaskablePPO training / fine-tune
    python -m agents.district.train_ppo        # config-driven trainer
    python -m agents.district.train_curriculum # opponent curriculum

See docs/rl_training_system.md.
"""

import sys
import subprocess

if __name__ == '__main__':
    print("=" * 70)
    print("RTS AI TRAINING — district tier")
    print("=" * 70)
    cmd = [sys.executable, '-m', 'agents.district.train_ppo'] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))
