"""
Guards on the district agent's action space.

`action_masks()` looks its maps up by `ActionType` member name, so an entry whose
key is not an ActionType member is unreachable: it reads like a supported action
while doing nothing. This is how BUILD_WAREHOUSE / BUILD_DORMITORY /
BUILD_WATCHTOWER / BUILD_DEFENSIVE_WALL sat in the map without existing.
"""

import inspect
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.district.env import RealTimeRTSEnv
from simulator.actions import ActionType


def _mask_action_names() -> set:
    source = inspect.getsource(RealTimeRTSEnv.action_masks)
    return set(re.findall(r"'((?:BUILD|TRAIN)_[A-Z_]+)'", source))


def test_action_masks_reference_only_real_action_types():
    """No dead entries: every name the mask keys on must exist in ActionType."""
    unknown = _mask_action_names() - {a.name for a in ActionType}
    assert not unknown, f"action_masks() maps unreachable actions: {sorted(unknown)}"


def test_every_buildable_and_trainable_action_is_masked():
    """The converse: no ActionType silently falls through to `else: mask = False`."""
    covered = _mask_action_names()
    declared = {a.name for a in ActionType if a.name.startswith(('BUILD_', 'TRAIN_'))}
    assert declared - covered == set()


def test_action_space_matches_action_type():
    env = RealTimeRTSEnv(opponent_type='normal')
    assert env.action_space.n == len(list(ActionType))
    assert len(env.action_masks()) == len(list(ActionType))


def test_do_nothing_is_always_available():
    """The mask must never be all-False, or MaskablePPO has nothing to sample."""
    env = RealTimeRTSEnv(opponent_type='normal')
    env.reset(seed=0)
    mask = env.action_masks()
    assert mask[list(ActionType).index(ActionType.DO_NOTHING)]
