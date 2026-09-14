"""
Game Data Loader
Loads game configuration from game_data/ JSON files and adapts them for the ML simulator.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any


class GameDataLoader:
    """Loads and adapts game data from JSON files for the ML simulator."""

    def __init__(self, game_data_dir: str = None):
        """
        Initialize game data loader.

        Args:
            game_data_dir: Path to game_data directory. If None, auto-detects.
        """
        if game_data_dir is None:
            # Auto-detect: rl/simulator -> project_root/game_data
            current_file = Path(__file__)
            rl_dir = current_file.parent.parent
            project_root = rl_dir.parent
            game_data_dir = project_root / "game_data"

        self.game_data_dir = Path(game_data_dir)

        if not self.game_data_dir.exists():
            raise FileNotFoundError(
                f"Game data directory not found: {self.game_data_dir}\n"
                f"Expected structure: project_root/game_data/*.json"
            )

        # Load all JSON files
        self.resource_data = self._load_json('resource.json')
        self.gameplay_data = self._load_json('gameplay.json')
        self.building_data = self._load_json('building.json')
        # time.json, person.json, map.json available if needed

    def _load_json(self, filename: str) -> Dict[str, Any]:
        """Load a JSON file from game_data directory."""
        filepath = self.game_data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Game data file not found: {filepath}")

        with open(filepath, 'r') as f:
            return json.load(f)

    def get_building_costs(self) -> Dict[str, Dict[str, int]]:
        """
        Extract building costs from resource.json.

        Returns:
            Dict mapping building name to resource costs
            Example: {'farm': {'wood': 20, 'stone': 10}}
        """
        construction_reqs = self.resource_data.get('construction_requirements', {})

        costs = {}
        for building_name, requirements in construction_reqs.items():
            # Handle multi-level buildings (like warehouse)
            if isinstance(requirements, dict) and 'level_1' in requirements:
                # For multi-level, use level_1 costs
                costs[building_name] = requirements['level_1']
            elif 'default' not in requirements:  # Skip 'default' entry
                costs[building_name] = requirements

        # Map JSON names to simulator names
        name_mapping = {
            'pottery_workshop': 'pottery',
            'carpentry_workshop': 'carpentry',
            'blacksmith_workshop': 'blacksmith',
            'stone_cutter_workshop': 'stone_cutter',
        }

        # Apply name mapping
        for json_name, sim_name in name_mapping.items():
            if json_name in costs:
                costs[sim_name] = costs.pop(json_name)

        return costs

    def get_starting_resources(self) -> Dict[str, int]:
        """
        Extract starting resources from gameplay.json.

        Returns:
            Dict mapping resource name to starting amount
        """
        return self.gameplay_data.get('starting_resources', {})

    def get_production_recipes(self) -> Dict[str, Dict[str, Any]]:
        """
        Extract production recipes from resource.json.
        Adapts to simulator's RESOURCE_PROCESSING format.

        Returns:
            Dict mapping building to {input, output} resources
        """
        recipes = self.resource_data.get('recipes', {})

        processed = {}
        for building, recipe in recipes.items():
            # Separate inputs and outputs
            inputs = {}
            outputs = {}

            for key, value in recipe.items():
                if key.startswith('input_'):
                    resource = key.replace('input_', '')
                    inputs[resource] = value
                elif key.startswith('output_'):
                    resource = key.replace('output_', '')
                    outputs[resource] = value

            if inputs or outputs:
                processed[building] = {}
                if inputs:
                    processed[building]['input'] = inputs
                if outputs:
                    processed[building]['output'] = outputs

        return processed

    def get_population_capacity_per_building(self) -> Dict[str, int]:
        """
        Extract population capacity per housing type.

        Returns:
            Dict mapping building to population capacity
        """
        pop_data = self.gameplay_data.get('population', {})
        return {
            'house': pop_data.get('capacity_per_house', 4),
            'dormitory': pop_data.get('capacity_per_dormitory', 8),
        }

    def get_building_build_times(self) -> Dict[str, float]:
        """
        Extract building construction times from building.json.
        Times are in seconds, will be converted to minutes in config.

        Returns:
            Dict mapping building name to build time in seconds
        """
        construction_time = self.building_data.get('construction_time', {})

        # Flatten the categorized structure
        times = {}
        for category in ['quick_builds', 'standard_builds', 'complex_builds', 'advanced_builds']:
            if category in construction_time:
                times.update(construction_time[category])

        return times

    def get_unit_training_data(self) -> Dict[str, Any]:
        """
        Extract unit training times, costs, and strength from gameplay.json.

        Returns:
            Dict with 'times', 'costs', 'strength' keys
        """
        military = self.gameplay_data.get('military', {})

        return {
            'times': military.get('training_time', {}),
            'costs': military.get('training_cost', {}),
            'strength': military.get('unit_strength', {}),
        }


# Global loader instance
_loader = None


def get_loader() -> GameDataLoader:
    """Get or create global game data loader instance."""
    global _loader
    if _loader is None:
        _loader = GameDataLoader()
    return _loader


def load_building_costs() -> Dict[str, Dict[str, int]]:
    """Load building costs from game_data/resource.json."""
    return get_loader().get_building_costs()


def load_starting_resources() -> Dict[str, int]:
    """Load starting resources from game_data/gameplay.json."""
    return get_loader().get_starting_resources()


def load_production_recipes() -> Dict[str, Dict[str, Any]]:
    """Load production recipes from game_data/resource.json."""
    return get_loader().get_production_recipes()


def load_population_capacities() -> Dict[str, int]:
    """Load population capacities from game_data/gameplay.json."""
    return get_loader().get_population_capacity_per_building()


def load_building_build_times() -> Dict[str, float]:
    """Load building construction times from game_data/building.json."""
    return get_loader().get_building_build_times()


def load_unit_training_data() -> Dict[str, Any]:
    """Load unit training data from game_data/gameplay.json."""
    return get_loader().get_unit_training_data()


if __name__ == '__main__':
    # Test the loader
    print("Testing game_data_loader...")
    print("=" * 60)

    loader = GameDataLoader()

    print("\n✅ Building Costs:")
    costs = loader.get_building_costs()
    for building, cost in list(costs.items())[:5]:
        print(f"  {building}: {cost}")

    print(f"\n✅ Starting Resources:")
    resources = loader.get_starting_resources()
    print(f"  {resources}")

    print(f"\n✅ Production Recipes:")
    recipes = loader.get_production_recipes()
    for building, recipe in list(recipes.items())[:3]:
        print(f"  {building}: {recipe}")

    print(f"\n✅ Population Capacities:")
    caps = loader.get_population_capacity_per_building()
    print(f"  {caps}")

    print(f"\n✅ Building Build Times (seconds):")
    build_times = loader.get_building_build_times()
    for building, time in list(build_times.items())[:5]:
        print(f"  {building}: {time}s ({time/60:.1f}min)")

    print(f"\n✅ Unit Training Data:")
    unit_data = loader.get_unit_training_data()
    print(f"  Training times: {unit_data['times']}")
    print(f"  Training costs: {unit_data['costs']}")
    print(f"  Unit strength: {unit_data['strength']}")
