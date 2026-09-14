"""
Achievement Rewards System for AI Training
Maps achievements to milestone rewards for reinforcement learning
"""

import json
import os
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field


@dataclass
class AchievementReward:
    """Defines a single achievement and its training reward."""
    id: str
    name: str
    description: str
    category: str
    difficulty: str
    reward_points: float
    condition: Dict
    unlocked: bool = False


class AchievementRewardMapper:
    """
    Maps achievements to AI training rewards based on difficulty.
    Provides methods to check achievement completion and calculate rewards.
    """

    # Reward multipliers based on difficulty
    DIFFICULTY_REWARDS = {
        'beginner': 10.0,
        'intermediate': 25.0,
        'advanced': 50.0,
        'expert': 100.0,
        'master': 200.0
    }

    # Category bonuses (added to base reward)
    CATEGORY_BONUSES = {
        'economy': 10.0,
        'building': 15.0,
        'population': 10.0,
        'military': 30.0,
        'victory': 100.0,
        'efficiency': 40.0,
        'challenge': 60.0,
        'collection': 50.0
    }

    def __init__(self, achievements_path: str = None):
        """
        Initialize achievement mapper.

        Args:
            achievements_path: Path to achievements.json file. If None, uses default path.
        """
        if achievements_path is None:
            # Default path relative to this file
            # File is at: rl/simulator/achievements_rewards.py
            # Need to go up 3 levels to reach project root
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            achievements_path = os.path.join(base_dir, 'game_data', 'achievements.json')

        self.achievements: Dict[str, AchievementReward] = {}
        self.unlocked_achievements: Set[str] = set()
        self._load_achievements(achievements_path)

    def _load_achievements(self, path: str):
        """Load achievements from JSON and calculate rewards."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)

            achievements_data = data.get('achievements', {})

            for category, achievements in achievements_data.items():
                for achievement_id, achievement in achievements.items():
                    # Calculate reward based on difficulty and category
                    difficulty = achievement.get('difficulty', 'beginner')
                    base_reward = self.DIFFICULTY_REWARDS.get(difficulty, 5.0)
                    category_bonus = self.CATEGORY_BONUSES.get(category, 0.0)
                    total_reward = base_reward + category_bonus

                    # Create achievement reward object
                    self.achievements[achievement_id] = AchievementReward(
                        id=achievement_id,
                        name=achievement.get('name', ''),
                        description=achievement.get('description', ''),
                        category=category,
                        difficulty=difficulty,
                        reward_points=total_reward,
                        condition=achievement.get('condition', {})
                    )

        except FileNotFoundError:
            print(f"Warning: Achievements file not found at {path}")
        except json.JSONDecodeError as e:
            print(f"Warning: Error parsing achievements JSON: {e}")

    def check_achievement(self, achievement_id: str, game_state, game_stats: Dict) -> bool:
        """
        Check if an achievement condition is met.

        Args:
            achievement_id: ID of the achievement to check
            game_state: Current game state object
            game_stats: Dictionary with game statistics (battles_won, resources_collected, etc.)

        Returns:
            True if achievement unlocked, False otherwise
        """
        if achievement_id not in self.achievements:
            return False

        if achievement_id in self.unlocked_achievements:
            return False  # Already unlocked

        achievement = self.achievements[achievement_id]
        condition = achievement.condition
        condition_type = condition.get('type', '')

        # Get faction (assume faction_id 0 for single-player AI training)
        faction = game_state.get_faction(0) if hasattr(game_state, 'get_faction') else None
        if faction is None:
            return False

        # Check different condition types
        if condition_type == 'resource_amount':
            resource = condition.get('resource', '')
            amount = condition.get('amount', 0)
            if resource == 'any':
                return any(faction.get_resource(r) >= amount for r in faction.resources)
            else:
                return faction.get_resource(resource) >= amount

        elif condition_type == 'resources_amount':
            resources = condition.get('resources', [])
            amount = condition.get('amount', 0)
            return all(faction.get_resource(r) >= amount for r in resources)

        elif condition_type == 'resource_collected':
            resource = condition.get('resource', '')
            amount = condition.get('amount', 0)
            return game_stats.get(f'{resource}_collected', 0) >= amount

        elif condition_type == 'buildings_completed':
            amount = condition.get('amount', 0)
            return sum(faction.buildings.values()) >= amount

        elif condition_type == 'building_count':
            building = condition.get('building', '')
            amount = condition.get('amount', 0)
            return faction.get_building_count(building) >= amount

        elif condition_type == 'buildings_owned':
            buildings = condition.get('buildings', [])
            min_count = condition.get('min_count', 1)
            return all(faction.get_building_count(b) >= min_count for b in buildings)

        elif condition_type == 'population':
            amount = condition.get('amount', 0)
            return faction.population >= amount

        elif condition_type == 'capacity_surplus':
            amount = condition.get('amount', 0)
            return (faction.population_capacity - faction.population) >= amount

        elif condition_type == 'unit_count':
            amount = condition.get('amount', 0)
            return sum(faction.units.values()) >= amount

        elif condition_type == 'unit_type_count':
            unit = condition.get('unit', '')
            amount = condition.get('amount', 0)
            return faction.get_unit_count(unit) >= amount

        elif condition_type == 'units_owned':
            units = condition.get('units', [])
            min_count = condition.get('min_count', 1)
            return all(faction.get_unit_count(u) >= min_count for u in units)

        elif condition_type == 'military_strength':
            amount = condition.get('amount', 0)
            return faction.military_strength >= amount

        elif condition_type == 'games_won':
            amount = condition.get('amount', 0)
            return game_stats.get('games_won', 0) >= amount

        elif condition_type == 'battles_won':
            amount = condition.get('amount', 0)
            return game_stats.get('battles_won', 0) >= amount

        elif condition_type == 'balanced_stats':
            pop = condition.get('population', 0)
            buildings = condition.get('buildings', 0)
            military = condition.get('military', 0)
            return (faction.population >= pop and
                    sum(faction.buildings.values()) >= buildings and
                    sum(faction.units.values()) >= military)

        elif condition_type == 'win_in_time':
            time_seconds = condition.get('time_seconds', 0)
            return (game_state.game_over and
                    game_state.winner == 0 and
                    game_stats.get('game_time', float('inf')) <= time_seconds)

        # Add more condition types as needed...

        return False

    def check_all_achievements(self, game_state, game_stats: Dict) -> List[AchievementReward]:
        """
        Check all achievements and return newly unlocked ones.

        Args:
            game_state: Current game state
            game_stats: Game statistics dictionary

        Returns:
            List of newly unlocked achievements
        """
        newly_unlocked = []

        for achievement_id in self.achievements:
            if self.check_achievement(achievement_id, game_state, game_stats):
                self.unlocked_achievements.add(achievement_id)
                achievement = self.achievements[achievement_id]
                achievement.unlocked = True
                newly_unlocked.append(achievement)

        return newly_unlocked

    def get_reward_for_achievements(self, achievements: List[AchievementReward]) -> float:
        """
        Calculate total reward points for a list of achievements.

        Args:
            achievements: List of unlocked achievements

        Returns:
            Total reward points
        """
        return sum(a.reward_points for a in achievements)

    def reset(self):
        """Reset all unlocked achievements (for new training episode)."""
        self.unlocked_achievements.clear()
        for achievement in self.achievements.values():
            achievement.unlocked = False

    def get_achievement_stats(self) -> Dict:
        """Get statistics about achievements."""
        total = len(self.achievements)
        unlocked = len(self.unlocked_achievements)

        by_category = {}
        by_difficulty = {}

        for achievement in self.achievements.values():
            # Count by category
            cat = achievement.category
            if cat not in by_category:
                by_category[cat] = {'total': 0, 'unlocked': 0}
            by_category[cat]['total'] += 1
            if achievement.unlocked:
                by_category[cat]['unlocked'] += 1

            # Count by difficulty
            diff = achievement.difficulty
            if diff not in by_difficulty:
                by_difficulty[diff] = {'total': 0, 'unlocked': 0}
            by_difficulty[diff]['total'] += 1
            if achievement.unlocked:
                by_difficulty[diff]['unlocked'] += 1

        return {
            'total': total,
            'unlocked': unlocked,
            'progress': unlocked / total if total > 0 else 0,
            'by_category': by_category,
            'by_difficulty': by_difficulty
        }

    def get_total_possible_reward(self) -> float:
        """Get the total possible reward points from all achievements."""
        return sum(a.reward_points for a in self.achievements.values())

    def get_unlocked_reward_total(self) -> float:
        """Get total reward points from unlocked achievements."""
        return sum(a.reward_points for a in self.achievements.values() if a.unlocked)


# Pre-configured reward mapper instance
_default_mapper = None

def get_achievement_mapper() -> AchievementRewardMapper:
    """Get the default achievement mapper (singleton)."""
    global _default_mapper
    if _default_mapper is None:
        _default_mapper = AchievementRewardMapper()
    return _default_mapper


# Quick reference: Achievement IDs organized by category
ACHIEVEMENT_IDS = {
    'economy': [
        'first_harvest', 'woodcutter', 'stone_mason', 'master_gatherer',
        'resource_tycoon', 'iron_age', 'golden_empire', 'self_sufficient',
        'industrial_revolution'
    ],
    'building': [
        'first_foundation', 'architect', 'master_builder', 'metropolis',
        'diversified_economy', 'production_chain_master', 'weapons_manufacturer',
        'construction_speed'
    ],
    'population': [
        'village_founded', 'thriving_town', 'bustling_city', 'megacity',
        'housing_authority', 'no_homelessness', 'population_boom'
    ],
    'military': [
        'first_blood', 'standing_army', 'military_power', 'unstoppable_force',
        'archer_division', 'infantry_regiment', 'combined_arms', 'first_strike',
        'victorious', 'warlord', 'conqueror', 'defender'
    ],
    'victory': [
        'first_victory', 'dominant_victory', 'economic_victory', 'speed_runner',
        'untouchable', 'perfect_game'
    ],
    'efficiency': [
        'efficient_builder', 'no_waste', 'balanced_growth', 'sustainable_economy',
        'military_industrial_complex', 'resource_efficiency', 'fast_expansion'
    ],
    'challenge': [
        'underdog', 'comeback_king', 'resource_crisis', 'minimal_army',
        'pacifist_almost', 'builders_challenge'
    ],
    'collection': [
        'jack_of_all_trades', 'specialized_economy', 'complete_arsenal',
        'research_complete', 'veteran_player', 'master_strategist',
        'legendary_commander'
    ]
}


if __name__ == '__main__':
    # Test and display achievement rewards
    mapper = AchievementRewardMapper()

    print("=== Achievement Rewards System ===\n")
    print(f"Total Achievements: {len(mapper.achievements)}")
    print(f"Total Possible Reward: {mapper.get_total_possible_reward():.1f} points\n")

    print("Achievements by Category:")
    for category, achievement_ids in ACHIEVEMENT_IDS.items():
        print(f"\n{category.upper()}:")
        for achievement_id in achievement_ids:
            if achievement_id in mapper.achievements:
                ach = mapper.achievements[achievement_id]
                print(f"  [{ach.difficulty:12s}] {ach.name:30s} = {ach.reward_points:5.1f} pts")
