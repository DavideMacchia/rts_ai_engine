"""
Scripted opponents module for training.

This module contains AI opponents for the agent to train against.

Structure:
- base.py: ScriptedOpponent base class
- aggressive.py: AggressiveBot — the combat bot (DEPRECATED as a teacher; kept for the
  combat-oracle tests and the future military camp tier)
- settler.py: SettlerBot — the current teacher for the civil district tier (D32)
- testing.py: TestingBot — balanced economy bot, test fixture only
- registry.py: Opponent pool and helper functions
"""

# Import base class
from .base import ScriptedOpponent

# Import opponent bots (NormalBot is a deprecated alias of TestingBot)
from .testing import TestingBot, NormalBot
from .aggressive import AggressiveBot

# Import registry functions
from .registry import (
    get_opponent,
    get_curriculum_sequence,
    get_all_difficulties,
    get_opponent_info,
    OPPONENT_POOL
)

__all__ = [
    # Base class
    'ScriptedOpponent',

    # Opponent bots
    'TestingBot',
    'NormalBot',       # deprecated alias
    'AggressiveBot',

    # Registry functions
    'get_opponent',
    'get_curriculum_sequence',
    'get_all_difficulties',
    'get_opponent_info',
    'OPPONENT_POOL',
]
