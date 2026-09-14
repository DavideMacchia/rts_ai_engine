"""
Configuration Loader for ML Training
Loads configuration from JSON files instead of hardcoded values
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigLoader:
    """Load and provide access to configuration from JSON files"""

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration loader

        Args:
            config_dir: Directory containing config files. If None, looks in default locations.
        """
        if config_dir is None:
            # Calculate paths relative to this file's location
            # This file is in: rl/config/config_loader.py
            # We want to reach: src/assets/constants/

            current_file = Path(__file__)  # rl/config/config_loader.py
            rl_dir = current_file.parent.parent  # rl/
            project_root = rl_dir.parent  # project root

            # Try multiple locations in order of preference
            # Prioritize rl/config/ (where configs are now stored)
            possible_paths = [
                current_file.parent,  # rl/config/ (PRIMARY LOCATION)
                Path.cwd() / "config",  # ./config from current directory
                project_root / "config",  # project_root/config/
            ]

            for path in possible_paths:
                if path.exists() and path.is_dir():
                    config_dir = str(path)
                    break

            if config_dir is None:
                # Fallback to same directory as this file
                config_dir = str(current_file.parent)

        self.config_dir = Path(config_dir)
        self._configs: Dict[str, Dict[str, Any]] = {}

    def load(self, config_name: str) -> Dict[str, Any]:
        """
        Load a configuration file

        Args:
            config_name: Name of config file (without .json extension)

        Returns:
            Dict containing configuration
        """
        if config_name in self._configs:
            return self._configs[config_name]

        config_path = self.config_dir / f"{config_name}.json"

        if not config_path.exists():
            # Calculate paths for error message
            current_file = Path(__file__)
            rl_config = current_file.parent

            # Provide helpful error message with search paths
            searched_paths = [
                ml_training_config / f"{config_name}.json",
                Path.cwd() / "config" / f"{config_name}.json",
            ]

            error_msg = f"❌ Config file not found: {config_path}\n"
            error_msg += f"   Current search directory: {self.config_dir}\n\n"
            error_msg += f"Searched in the following locations:\n"
            for i, path in enumerate(searched_paths, 1):
                exists = "✓ EXISTS" if path.exists() else "✗ not found"
                error_msg += f"  {i}. {path}\n     [{exists}]\n"

            error_msg += f"\n💡 To fix this, place {config_name}.json in:\n"
            error_msg += f"   {rl_config}/\n"
            error_msg += f"   (Recommended: rl/config/{config_name}.json)\n"

            raise FileNotFoundError(error_msg)

        with open(config_path, 'r') as f:
            config = json.load(f)

        self._configs[config_name] = config
        return config

    def get(self, config_name: str, *keys: str, default: Any = None) -> Any:
        """
        Get a nested configuration value

        Args:
            config_name: Name of config file
            *keys: Nested keys to traverse
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            loader.get('ml_training_config', 'training', 'total_timesteps')
        """
        try:
            config = self.load(config_name)
            value = config
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            raise KeyError(f"Config key not found: {config_name}.{'.'.join(keys)}")

    def get_section(self, config_name: str, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section

        Args:
            config_name: Name of config file
            section: Section name

        Returns:
            Dict containing the section
        """
        config = self.load(config_name)
        return config.get(section, {})


# Global instance for convenience
_global_loader: Optional[ConfigLoader] = None


def get_config_loader(config_dir: Optional[str] = None) -> ConfigLoader:
    """
    Get global configuration loader instance

    Args:
        config_dir: Optional directory containing config files

    Returns:
        ConfigLoader instance
    """
    global _global_loader
    if _global_loader is None:
        _global_loader = ConfigLoader(config_dir)
    return _global_loader


def load_training_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load ML training configuration

    Args:
        config_path: Optional path to specific config file

    Returns:
        Training configuration dict
    """
    if config_path:
        with open(config_path, 'r') as f:
            return json.load(f)

    loader = get_config_loader()
    return loader.load('ml_training_config')


def load_transformer_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load transformer architecture configuration

    Args:
        config_path: Optional path to specific config file

    Returns:
        Transformer configuration dict
    """
    if config_path:
        with open(config_path, 'r') as f:
            return json.load(f)

    loader = get_config_loader()
    return loader.load('transformer_config')


def load_env_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load environment configuration

    Args:
        config_path: Optional path to specific config file

    Returns:
        Environment configuration dict
    """
    if config_path:
        with open(config_path, 'r') as f:
            cfg = json.load(f)
    else:
        cfg = get_config_loader().load('env_config')

    _apply_time_scale(cfg)
    return cfg


def _apply_time_scale(env_cfg: Dict[str, Any]) -> None:
    """Overwrite the JSON's time values with the single source of truth in config.py.

    The episode length and decision interval are DERIVED from the adult-life anchor
    (config.py TIME SCALE). Keeping a second copy in env_config.json meant a retune had
    to be mirrored by hand, and a stale copy silently trained the agent at the wrong
    scale. The JSON keeps everything non-temporal (feature_keys, spaces); time comes
    from here. Imported lazily to avoid an import cycle (config.py imports loaders).
    """
    from simulator.config import EPISODE_SECONDS, DECISION_INTERVAL_SECONDS

    env = env_cfg.get('environment', {})
    env['max_game_time'] = EPISODE_SECONDS
    env['decision_interval'] = DECISION_INTERVAL_SECONDS


def load_gameplay_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load gameplay configuration (game phases, normalization constants, etc.)

    Args:
        config_path: Optional path to specific config file

    Returns:
        Gameplay configuration dict
    """
    if config_path:
        with open(config_path, 'r') as f:
            return json.load(f)

    loader = get_config_loader()
    return loader.load('gameplay_config')