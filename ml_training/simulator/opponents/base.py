"""
Base class for scripted opponents.
All opponent bots inherit from ScriptedOpponent.
"""

from ..actions import Action, ActionType
from ..game_state import Faction, GameState
from typing import Optional
import sys
import os

# Add path for config loader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config_loader import load_gameplay_config

# Load gameplay constants
try:
    _GAMEPLAY_CONFIG = load_gameplay_config()
    GAME_PHASES = _GAMEPLAY_CONFIG['game_phases']
    OPPONENT_STRATEGY = _GAMEPLAY_CONFIG['opponent_strategy']
except Exception:
    # Fallback to hardcoded values if config not available
    GAME_PHASES = {'early_game_end': 7200, 'mid_game_end': 14400, 'late_game_start': 14400}
    OPPONENT_STRATEGY = {
        'detection_delay_seconds': 1800,
        'rush_threshold': {'military_ratio': 0.3, 'min_barracks': 1},
        'boom_threshold': {'min_buildings': 10, 'max_military_ratio': 0.1}
    }


class ScriptedOpponent:
    """Base class for rule-based opponents."""

    def __init__(self, faction_id: int):
        """
        Initialize scripted opponent.

        Args:
            faction_id: ID of the faction this bot controls
        """
        self.faction_id = faction_id
        self.name = "ScriptedOpponent"
        self.difficulty = "unknown"

    def act(self, game_state: GameState, game_time: float) -> Action:
        """
        Decide what action to take.

        Args:
            game_state: Current game state
            game_time: Current game time in seconds

        Returns:
            Action to execute
        """
        raise NotImplementedError("Subclasses must implement act()")

    def _get_faction(self, game_state: GameState) -> Optional[Faction]:
        """Get the faction this bot controls."""
        return game_state.get_faction(self.faction_id)

    def _get_opponent(self, game_state: GameState) -> Optional[Faction]:
        """Get the opponent faction."""
        opponent_id = 1 if self.faction_id == 0 else 0
        return game_state.get_faction(opponent_id)
