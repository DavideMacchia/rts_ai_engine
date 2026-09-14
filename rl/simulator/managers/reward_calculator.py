"""
Reward calculation for the RTS simulator.
Handles reward calculations for reinforcement learning.
"""

from typing import Dict
from ..game_state import GameState
from ..config import EPISODE_SECONDS

# Import achievement system
try:
    from ..achievements_rewards import AchievementRewardMapper
except ImportError:
    AchievementRewardMapper = None
    print("Warning: Achievement system not available")


class RewardCalculator:
    """Calculates rewards for RL training."""

    def __init__(self, reward_config: Dict, enable_achievements: bool = True):
        """
        Initialize reward calculator.

        Args:
            reward_config: Reward configuration dictionary
            enable_achievements: Whether to enable achievement-based rewards
        """
        self.reward_config = reward_config
        self.enable_achievements = enable_achievements and AchievementRewardMapper is not None

        # Initialize achievement system
        if self.enable_achievements:
            self.achievement_mapper = AchievementRewardMapper()
        else:
            self.achievement_mapper = None

        # Game statistics tracking for achievements
        self.game_stats = {
            'games_won': 0,
            'games_played': 0,
            'battles_won': 0,
            'attacks_launched': 0,
            'attacks_survived': 0,
            'warehouses_destroyed': 0,
            'units_trained': 0,
            'buildings_completed': 0,
            'game_time': 0.0,
            'max_population': 0,
            'max_military_strength': 0,
            'resources_collected': {},
            'buildings_lost': 0,
            'start_time': 0.0,
            'last_5_buildings_times': []
        }

        # Milestone tracking per faction (one-time rewards)
        self.milestones_achieved = {}

        # Previous-state tracking per faction (for delta-based rewards)
        self.prev_state = {}

    def _get_snapshot(self, faction) -> Dict:
        """Capture the metrics we compute deltas against."""
        return {
            'population': faction.population,
            'population_capacity': faction.population_capacity,
            'buildings': sum(faction.buildings.values()),
            'units': sum(faction.units.values()),
            'military_strength': faction.military_strength,
        }

    # NOTE: there is no `calculate_timeout_rewards`. Running out of time pays nothing.
    # It is a truncation, not an outcome — the value function bootstraps across it.
    # Paying for a timeout is what taught the agent to hoard an army and never attack.

    def calculate_rewards(self, game_state: GameState, game_time: float) -> Dict[int, float]:
        """
        Calculate rewards with balanced milestone and progress incentives.

        Philosophy:
        - Small continuous rewards for progress (building, resources, units)
        - Medium rewards for strategic milestones (first barracks, first attack)
        - Large rewards for winning/losing
        - Penalties for stagnation and poor decisions

        Args:
            game_state: Current game state
            game_time: Current game time in seconds

        Returns:
            Dict[int, float]: Rewards for each faction
        """
        cfg = self.reward_config
        rewards = {faction.id: 0.0 for faction in game_state.factions}

        for faction in game_state.factions:
            reward = 0.0

            opponent_id = 1 if faction.id == 0 else 0
            opponent = game_state.get_faction(opponent_id)

            # ===== VICTORY/DEFEAT =====
            # Only conquest sets a winner. A timeout leaves `winner` None and pays
            # nothing to anybody: it is a truncation, not a result.
            winner = game_state.winner
            if winner is not None:
                if winner == faction.id:
                    reward += cfg.get('victory_defeat', 'victory')
                    # Bonus for the hours saved against the EPISODE, not against a
                    # 200-hour MAX_TURNS the simulator never reaches — that made the
                    # "win fast" bonus a near-constant added to every victory.
                    hours_saved = (EPISODE_SECONDS - game_time) / 3600.0
                    time_bonus = hours_saved * cfg.get('victory_defeat', 'victory_time_bonus_per_hour_saved')
                    reward += max(0.0, time_bonus)
                elif faction.is_defeated():
                    reward += cfg.get('victory_defeat', 'defeat')

            # ===== DELTA-BASED PROGRESS =====
            # Reward positive CHANGES, not absolute state.
            # This removes the "idle and collect" exploit: doing nothing = ~0 reward.
            prev = self.prev_state.get(faction.id)
            curr = self._get_snapshot(faction)

            if prev is not None:
                # Economic deltas
                d_pop = curr['population'] - prev['population']
                if d_pop > 0:
                    reward += d_pop * cfg.get('economic_progress', 'population_growth')
                d_cap = curr['population_capacity'] - prev['population_capacity']
                if d_cap > 0:
                    reward += d_cap * cfg.get('economic_progress', 'capacity_growth')
                d_build = curr['buildings'] - prev['buildings']
                if d_build > 0:
                    reward += d_build * cfg.get('economic_progress', 'building_completed')

                # Military deltas
                d_units = curr['units'] - prev['units']
                if d_units > 0:
                    reward += d_units * cfg.get('military_progress', 'unit_trained')
                d_mil = curr['military_strength'] - prev['military_strength']
                if d_mil > 0:
                    reward += d_mil * cfg.get('military_progress', 'strength_gained')

            self.prev_state[faction.id] = curr

            # Small ongoing signal for units currently in training (progress toward army)
            units_in_training = len(faction.units_in_training)
            reward += units_in_training * cfg.get('military_progress', 'unit_in_training')

            # ===== MILITARY MILESTONES (one-time) =====
            fid = faction.id
            if fid not in self.milestones_achieved:
                self.milestones_achieved[fid] = set()

            milestones = self.milestones_achieved[fid]

            if 'first_barracks' not in milestones and faction.get_building_count('barracks') > 0:
                milestones.add('first_barracks')
                reward += cfg.get('military_milestones', 'first_barracks')

            if 'first_soldier' not in milestones and faction.get_unit_count('soldier') > 0:
                milestones.add('first_soldier')
                reward += cfg.get('military_milestones', 'first_unit')

            if 'first_archer' not in milestones and faction.get_unit_count('archer') > 0:
                milestones.add('first_archer')
                reward += cfg.get('military_milestones', 'first_unit')

            if 'army_3' not in milestones and sum(faction.units.values()) >= 3:
                milestones.add('army_3')
                reward += cfg.get('military_milestones', 'army_size_3')

            if 'army_5' not in milestones and sum(faction.units.values()) >= 5:
                milestones.add('army_5')
                reward += cfg.get('military_milestones', 'army_size_5')

            if 'military_superiority' not in milestones and opponent:
                if faction.military_strength > opponent.military_strength and faction.military_strength > 0:
                    milestones.add('military_superiority')
                    reward += cfg.get('military_milestones', 'military_superiority')

            # ===== PENALTIES =====
            # DO_NOTHING opportunity cost: small per-step penalty for having no
            # barracks late-game (encourages the military path without punishing
            # the build itself, since building takes many steps to complete).
            military_time = cfg.get('timing_thresholds', 'military_check')
            if game_time > military_time and faction.get_building_count('barracks') == 0:
                reward += cfg.get('penalties', 'no_barracks_after_check')

            stagnation_time = cfg.get('timing_thresholds', 'stagnation_check')
            if game_time > stagnation_time:
                if sum(faction.buildings.values()) == 0:
                    reward += cfg.get('penalties', 'stagnation_no_buildings_after_1h')

            if faction.population == 0 and not faction.is_defeated():
                reward += cfg.get('penalties', 'population_collapse')

            resource_time = cfg.get('timing_thresholds', 'resource_starvation_check')
            if game_time > resource_time:
                if faction.get_resource('wood') < 5 and faction.get_resource('stone') < 5:
                    reward += cfg.get('penalties', 'resource_starvation_after_1h')

            farm_count = faction.get_building_count('farm')
            farm_threshold = int(cfg.get('penalties', 'excessive_farms_threshold'))
            if farm_count > farm_threshold:
                reward += (farm_count - farm_threshold) * cfg.get('penalties', 'excessive_farms_per_extra')

            # ===== ACHIEVEMENT REWARDS =====
            if self.enable_achievements and self.achievement_mapper:
                # Update game stats
                self.game_stats['game_time'] = game_time
                self.game_stats['max_population'] = max(self.game_stats.get('max_population', 0), faction.population)
                self.game_stats['max_military_strength'] = max(self.game_stats.get('max_military_strength', 0), faction.military_strength)

                # Check for newly unlocked achievements
                newly_unlocked = self.achievement_mapper.check_all_achievements(game_state, self.game_stats)

                # Add achievement rewards
                achievement_reward = self.achievement_mapper.get_reward_for_achievements(newly_unlocked)
                if achievement_reward > 0:
                    reward += achievement_reward

            rewards[faction.id] = reward

        return rewards

    def update_stat(self, stat_name: str, value):
        """
        Update a game statistic.

        Args:
            stat_name: Name of the stat to update
            value: New value (will be added if numeric, set if other type)
        """
        if stat_name in self.game_stats:
            if isinstance(value, (int, float)) and isinstance(self.game_stats[stat_name], (int, float)):
                self.game_stats[stat_name] += value
            else:
                self.game_stats[stat_name] = value

    def record_building_completed(self):
        """Record that a building was completed."""
        self.game_stats['buildings_completed'] += 1

    def record_unit_trained(self):
        """Record that a unit was trained."""
        self.game_stats['units_trained'] += 1

    def record_attack_launched(self):
        """Record that an attack was launched."""
        self.game_stats['attacks_launched'] += 1

    def record_battle_won(self):
        """Record a battle victory."""
        self.game_stats['battles_won'] += 1

    def record_attack_survived(self):
        """Record surviving an enemy attack."""
        self.game_stats['attacks_survived'] += 1

    def record_warehouse_destroyed(self):
        """Record destroying an enemy warehouse."""
        self.game_stats['warehouses_destroyed'] += 1

    def record_building_lost(self):
        """Record losing a building."""
        self.game_stats['buildings_lost'] += 1

    def record_resource_collected(self, resource: str, amount: int):
        """
        Record resources collected.

        Args:
            resource: Name of the resource
            amount: Amount collected
        """
        key = f'{resource}_collected'
        if key not in self.game_stats:
            self.game_stats[key] = 0
        self.game_stats[key] += amount

    def record_game_completed(self, won: bool):
        """
        Record a completed game.

        Args:
            won: Whether the game was won
        """
        self.game_stats['games_played'] += 1
        if won:
            self.game_stats['games_won'] += 1

    def reset_episode(self):
        """Reset statistics for a new training episode."""
        # Reset per-episode stats but keep cumulative stats
        cumulative_stats = {
            'games_won': self.game_stats.get('games_won', 0),
            'games_played': self.game_stats.get('games_played', 0),
            'battles_won': self.game_stats.get('battles_won', 0),
        }

        self.game_stats = {
            'games_won': cumulative_stats['games_won'],
            'games_played': cumulative_stats['games_played'],
            'battles_won': cumulative_stats['battles_won'],
            'attacks_launched': 0,
            'attacks_survived': 0,
            'warehouses_destroyed': 0,
            'units_trained': 0,
            'buildings_completed': 0,
            'game_time': 0.0,
            'max_population': 0,
            'max_military_strength': 0,
            'resources_collected': {},
            'buildings_lost': 0,
            'start_time': 0.0,
            'last_5_buildings_times': []
        }

        # Reset milestones and delta-state
        self.milestones_achieved = {}
        self.prev_state = {}

        # Reset achievement tracker for new episode
        if self.enable_achievements and self.achievement_mapper:
            self.achievement_mapper.reset()

    def get_achievement_stats(self) -> Dict:
        """Get achievement statistics."""
        if self.enable_achievements and self.achievement_mapper:
            return self.achievement_mapper.get_achievement_stats()
        return {}

    def get_unlocked_achievements(self):
        """Get list of unlocked achievements."""
        if self.enable_achievements and self.achievement_mapper:
            return [ach for ach in self.achievement_mapper.achievements.values() if ach.unlocked]
        return []
