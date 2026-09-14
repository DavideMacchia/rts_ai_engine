"""
Opponent registry.

- the `SettlerBot` ('settler'): the CURRENT teacher, for the civil district tier (D32).
  It builds a settlement and never fights. This is what the civil agent trains against.
- the COMBAT bot (`AggressiveBot`) at three difficulties: 'easy' / 'medium' / 'hard',
  differing only in how long they wait before attacking. DEPRECATED as a teacher: the
  focus is the civil tier, not the military one. Kept functional for the combat-oracle
  tests and for the future military camp tier — see aggressive.py.
- the 'testing' bot (`TestingBot`), a balanced economy bot kept only as a test fixture.

Difficulty names dropped the redundant 'aggressive_' prefix (every combat bot is
aggressive). The old names ('normal', 'aggressive', 'aggressive_hard', …) are still
accepted as deprecated aliases so existing tests and commands keep working.
"""

from .base import ScriptedOpponent
from .testing import TestingBot
from .aggressive import AggressiveBot
from .settler import SettlerBot
from typing import Dict, Type, List, Tuple, Callable


# Non-parameterised bots (default constructors).
OPPONENT_POOL: Dict[str, Type[ScriptedOpponent]] = {
    'testing': TestingBot,
    # The expert for the CIVIL district tier (D32): it builds a settlement and never fights.
    # War belongs to the camp tier and its own model.
    'settler': SettlerBot,
}

# The combat curriculum: one AggressiveBot per difficulty. Easier = attacks later, and
# hoards a bigger army before it will move at all. DEPRECATED as a training path (see the
# module docstring); kept functional for the combat-oracle tests and the future camp tier.
#
# `attack_army_strength` is a floor on marching, not a reason to march: since D25 the
# commit decision belongs to the arrival forecast. A HIGH floor is what makes a bot easy
# — it sits on an army instead of using it, and gives the agent time. 'hard' has no floor:
# the forecast has the only say, so it will raid an undefended district with two soldiers.
COMBAT_CURRICULUM: Dict[str, Callable[[int], ScriptedOpponent]] = {
    'easy':   lambda fid: AggressiveBot(fid, min_attack_time=1200.0, attack_army_strength=4.0),
    'medium': lambda fid: AggressiveBot(fid, min_attack_time=800.0,  attack_army_strength=2.0),
    'hard':   lambda fid: AggressiveBot(fid, min_attack_time=500.0,  attack_army_strength=0.0),
}

# Deprecated name -> current name. Kept so old tests/commands don't break.
_DEPRECATED_ALIASES: Dict[str, str] = {
    'normal': 'testing',
    'aggressive': 'hard',
    'aggressive_easy': 'easy',
    'aggressive_medium': 'medium',
    'aggressive_hard': 'hard',
}


def _resolve(difficulty: str) -> str:
    return _DEPRECATED_ALIASES.get(difficulty, difficulty)


def get_opponent(difficulty: str, faction_id: int) -> ScriptedOpponent:
    """
    Get a scripted opponent of the specified difficulty.

    Args:
        difficulty: 'easy' | 'medium' | 'hard' (combat), or 'testing' (fixture).
                    Old names ('normal', 'aggressive', 'aggressive_hard', …) still work.
        faction_id: ID of the faction the bot should control
    """
    key = _resolve(difficulty)
    if key in COMBAT_CURRICULUM:
        return COMBAT_CURRICULUM[key](faction_id)
    bot_class = OPPONENT_POOL.get(key, TestingBot)
    return bot_class(faction_id)


def get_curriculum_sequence() -> List[Tuple[str, float, int]]:
    """The COMBAT curriculum: start against a real, gentle threat and ramp up.

    DEPRECATED for the current focus. This trains military play against the AggressiveBot;
    the civil district tier (D32) trains against the SettlerBot instead and does not use
    this sequence. Kept for the future military camp tier.
    """
    return [
        ('easy', 0.60, 500000),    # attacks late (1200s) — gentle but real
        ('medium', 0.55, 500000),  # attacks at 800s
        ('hard', 0.50, 500000),    # full ramping threat (500s)
    ]


def get_all_difficulties() -> List[str]:
    """Current (non-deprecated) difficulty names."""
    return list(OPPONENT_POOL.keys()) + list(COMBAT_CURRICULUM.keys())


def get_opponent_info(difficulty: str) -> Dict[str, str]:
    """Get name, difficulty, and a description for an opponent."""
    bot = get_opponent(difficulty, faction_id=0)

    descriptions = {
        'testing': 'Balanced economy bot. Never rushes; a test fixture, not a curriculum stage.',
        'easy': 'Combat bot, EASY: only attacks after 1200s. Lots of time for the agent to prepare.',
        'medium': 'Combat bot, MEDIUM: attacks after 800s.',
        'hard': 'Combat bot, HARD: attacks after 500s (full ramping threat).',
    }

    return {
        'name': bot.name,
        'difficulty': bot.difficulty,
        'description': descriptions.get(_resolve(difficulty), 'No description available')
    }
