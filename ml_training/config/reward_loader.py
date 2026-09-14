"""
Reward configuration loader
Loads reward values from JSON config file
"""

import json
import os
from typing import Dict, Any


class RewardConfig:
    """Loads and provides access to reward configuration"""

    def __init__(self, config_path: str = None):
        """
        Load reward configuration from JSON file

        Args:
            config_path: Path to JSON config file. If None, uses default.
        """
        if config_path is None:
            # Default to reward_config.json in same directory
            config_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(config_dir, 'reward_config.json')

        with open(config_path, 'r') as f:
            self._config = json.load(f)

        # Time thresholds are DERIVED from the adult-life anchor (config.py TIME SCALE),
        # not stored here, so a retune cannot leave a stale copy behind. Lazy import to
        # avoid an import cycle. See config_loader._apply_time_scale for the same pattern.
        from simulator.config import REWARD_TIMING_THRESHOLDS
        self._config.setdefault('timing_thresholds', {}).update(REWARD_TIMING_THRESHOLDS)

    def get(self, category: str, key: str, default: float = 0.0) -> float:
        """
        Get a reward value from config

        Args:
            category: Category name (e.g., 'action_validity')
            key: Key within category (e.g., 'valid_action')
            default: Default value if not found

        Returns:
            float: Reward value
        """
        try:
            return float(self._config[category][key])
        except (KeyError, TypeError):
            print(f"⚠️  Config key not found: {category}.{key}, using default {default}")
            return default

    def get_all(self, category: str) -> Dict[str, float]:
        """Get all values in a category"""
        return {k: v for k, v in self._config.get(category, {}).items()
                if not k.startswith('description')}

    @property
    def version(self) -> str:
        """Get config version"""
        return self._config.get('version', 'unknown')


# Global instance (loaded once)
_global_config = None


def get_reward_config(config_path: str = None) -> RewardConfig:
    """
    Get global reward config instance

    Args:
        config_path: Optional path to config file. Only used on first call.

    Returns:
        RewardConfig instance
    """
    global _global_config
    if _global_config is None:
        _global_config = RewardConfig(config_path)
    return _global_config


