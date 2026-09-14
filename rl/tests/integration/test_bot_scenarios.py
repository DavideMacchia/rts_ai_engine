"""
Scenario tests for NormalBot behavior across different game phases.
Tests early game, mid game, late game, and various combat scenarios.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from simulator.opponents import get_opponent
from simulator.game_state import GameState, Faction
from simulator.actions import ActionType


class TestEarlyGameBehavior:
    """Test NormalBot behavior in early game (0-10 minutes)."""

    def setup_method(self):
        """Setup early game scenario."""
        self.bot = get_opponent('normal', faction_id=1)
        self.game_state = GameState()
        self.game_state.factions = [Faction(id=0), Faction(id=1)]

        # Early game: limited resources, starting buildings
        faction = self.game_state.factions[1]
        faction.resources['wood'] = 300
        # Spears in the rack. Since D29 a soldier must be EQUIPPED: with an empty armoury
        # these bots cannot train anyone, and fall back to building an economy instead.
        faction.resources['wooden_weapon'] = 20
        # ...and the timber, bricks and dressed stone the buildings are now MADE of (D30):
        # the barracks is a timber building, the bakery and the dormitory are brick.
        faction.resources['wood_logs'] = 40
        faction.resources['clay_bricks'] = 40
        faction.resources['stone_bricks'] = 40
        faction.resources['stone'] = 200
        faction.resources['grain'] = 100
        faction.population = 10
        faction.population_capacity = 20
        faction.buildings['barracks'] = 0

    def test_early_game_prioritizes_economy(self):
        """Early game should focus on economy, not military."""
        game_time = 300  # 5 minutes

        actions = []
        for _ in range(10):
            action = self.bot.act(self.game_state, game_time)
            actions.append(action.action_type)

        # Count economic vs military actions
        economic = sum(1 for a in actions if a in [
            ActionType.BUILD_LUMBERYARD,
            ActionType.BUILD_QUARRY,
            ActionType.BUILD_FARM,
            ActionType.BUILD_WELL,
            ActionType.BUILD_HOUSE,
        ])

        military = sum(1 for a in actions if a in [
            ActionType.BUILD_BARRACKS,
            ActionType.TRAIN_SOLDIER,
            ActionType.TRAIN_ARCHER,
        ])

        assert economic > military, \
            f"Early game should prioritize economy over military (eco={economic}, mil={military})"

    def test_early_game_no_aggressive_attack(self):
        """Early game should not attack without military buildup."""
        game_time = 300  # 5 minutes

        # Even if we have some military, shouldn't attack in early game
        self.game_state.factions[1].military_strength = 10
        self.game_state.factions[0].military_strength = 5

        action = self.bot.act(self.game_state, game_time)

        actions = [self.bot.act(self.game_state, game_time) for _ in range(5)]

        # Unlikely to attack in early game even with advantage
        # (might attack if advantage is overwhelming, but generally builds eco first)
        attack_count = sum(1 for a in actions if a.action_type == ActionType.ATTACK)

        # Allow some attacks if advantage is huge, but shouldn't be constant attacking
        assert attack_count < 3, \
            "Early game should focus on building, not constant attacking"


class TestMidGameBehavior:
    """Test NormalBot behavior in mid game (10-30 minutes)."""

    def setup_method(self):
        """Setup mid game scenario."""
        self.bot = get_opponent('normal', faction_id=1)
        self.game_state = GameState()
        self.game_state.factions = [Faction(id=0), Faction(id=1)]

        # Mid game: good resources, established economy
        faction = self.game_state.factions[1]
        faction.resources['wood'] = 800
        # Spears in the rack. Since D29 a soldier must be EQUIPPED: with an empty armoury
        # these bots cannot train anyone, and fall back to building an economy instead.
        faction.resources['wooden_weapon'] = 20
        # ...and the timber, bricks and dressed stone the buildings are now MADE of (D30):
        # the barracks is a timber building, the bakery and the dormitory are brick.
        faction.resources['wood_logs'] = 40
        faction.resources['clay_bricks'] = 40
        faction.resources['stone_bricks'] = 40
        faction.resources['stone'] = 600
        faction.resources['grain'] = 400
        faction.population = 40
        faction.population_capacity = 60
        faction.buildings['lumberyard'] = 2
        faction.buildings['quarry'] = 2
        faction.buildings['farm'] = 3
        faction.buildings['house'] = 3

    def test_mid_game_builds_military(self):
        """Mid game should start building military infrastructure."""
        game_time = 1200  # 20 minutes

        actions = []
        for _ in range(20):
            action = self.bot.act(self.game_state, game_time)
            actions.append(action.action_type)

        # Should see some military actions
        has_military = any(a in [
            ActionType.BUILD_BARRACKS,
            ActionType.TRAIN_SOLDIER,
            ActionType.TRAIN_ARCHER
        ] for a in actions)

        assert has_military, \
            f"Mid game should include military actions, got: {[a.name for a in actions[:10]]}"

    def test_mid_game_balances_economy_and_military(self):
        """Mid game should balance economy and military."""
        game_time = 1200  # 20 minutes

        # Give bot a barracks so it can train units
        self.game_state.factions[1].buildings['barracks'] = 1

        actions = []
        for _ in range(20):
            action = self.bot.act(self.game_state, game_time)
            actions.append(action.action_type)

        economic = sum(1 for a in actions if a in [
            ActionType.BUILD_LUMBERYARD,
            ActionType.BUILD_QUARRY,
            ActionType.BUILD_FARM,
            ActionType.BUILD_WELL,
            ActionType.BUILD_HOUSE,
            ActionType.BUILD_MILL,
            ActionType.BUILD_BAKERY,
        ])

        military = sum(1 for a in actions if a in [
            ActionType.BUILD_BARRACKS,
            ActionType.TRAIN_SOLDIER,
            ActionType.TRAIN_ARCHER,
        ])

        # Should have both economic and military actions
        assert economic > 0, "Mid game should still build economy"
        assert military > 0, "Mid game should build military"

    @pytest.mark.skip(reason="Military reaction belongs to the CAMP tier now (D32): the civil district has no army, no war and no military actions in its space. Re-home this test when agents/camp/ exists — do not delete it, the behaviour it asks for is still wanted, just not HERE.")
    def test_mid_game_responds_to_threats(self):
        """Mid game should respond to enemy military buildup."""
        game_time = 1200  # 20 minutes

        # Enemy has built military
        self.game_state.factions[0].military_strength = 25
        self.game_state.factions[1].military_strength = 10

        action = self.bot.act(self.game_state, game_time)

        # Should prioritize military response
        assert action.action_type in [
            ActionType.BUILD_BARRACKS,
            ActionType.TRAIN_SOLDIER,
            ActionType.TRAIN_ARCHER,
        ], f"Should respond to threat with military, got {action.action_type.name}"


class TestLateGameBehavior:
    """Test NormalBot behavior in late game (30+ minutes)."""

    def setup_method(self):
        """Setup late game scenario."""
        self.bot = get_opponent('normal', faction_id=1)
        self.game_state = GameState()
        self.game_state.factions = [Faction(id=0), Faction(id=1)]

        # Late game: abundant resources, full economy
        faction = self.game_state.factions[1]
        faction.resources['wood'] = 2000
        # Spears in the rack. Since D29 a soldier must be EQUIPPED: with an empty armoury
        # these bots cannot train anyone, and fall back to building an economy instead.
        faction.resources['wooden_weapon'] = 20
        # ...and the timber, bricks and dressed stone the buildings are now MADE of (D30):
        # the barracks is a timber building, the bakery and the dormitory are brick.
        faction.resources['wood_logs'] = 40
        faction.resources['clay_bricks'] = 40
        faction.resources['stone_bricks'] = 40
        faction.resources['stone'] = 1500
        faction.resources['grain'] = 1000
        faction.population = 80
        faction.population_capacity = 100
        faction.buildings['lumberyard'] = 4
        faction.buildings['quarry'] = 4
        faction.buildings['farm'] = 6
        faction.buildings['house'] = 8
        faction.buildings['barracks'] = 2
        faction.military_strength = 25

    def test_late_game_attacks_when_strong(self):
        """Late game should attack when having military advantage."""
        game_time = 2400  # 40 minutes

        # Bot has good military, enemy is weaker
        self.game_state.factions[0].military_strength = 10
        self.game_state.factions[1].military_strength = 30

        action = self.bot.act(self.game_state, game_time)

        assert action.action_type == ActionType.ATTACK, \
            f"Late game with advantage should attack, got {action.action_type.name}"

    def test_late_game_continues_economy(self):
        """Late game should still expand economy when not attacking."""
        game_time = 2400  # 40 minutes

        # Equal military - shouldn't attack
        self.game_state.factions[0].military_strength = 25
        self.game_state.factions[1].military_strength = 25

        actions = []
        for _ in range(10):
            action = self.bot.act(self.game_state, game_time)
            actions.append(action.action_type)

        # Should continue building (not just do nothing)
        productive_actions = sum(1 for a in actions if a != ActionType.DO_NOTHING)

        assert productive_actions > 5, \
            "Late game should continue building when not attacking"


class TestCombatScenarios:
    """Test NormalBot behavior in various combat situations."""

    def setup_method(self):
        """Setup base combat scenario."""
        self.bot = get_opponent('normal', faction_id=1)
        self.game_state = GameState()
        self.game_state.factions = [Faction(id=0), Faction(id=1)]

        # Good resources for military
        faction = self.game_state.factions[1]
        faction.resources['wood'] = 1000
        # Spears in the rack. Since D29 a soldier must be EQUIPPED: with an empty armoury
        # these bots cannot train anyone, and fall back to building an economy instead.
        faction.resources['wooden_weapon'] = 20
        # ...and the timber, bricks and dressed stone the buildings are now MADE of (D30):
        # the barracks is a timber building, the bakery and the dormitory are brick.
        faction.resources['wood_logs'] = 40
        faction.resources['clay_bricks'] = 40
        faction.resources['stone_bricks'] = 40
        faction.resources['stone'] = 1000
        faction.resources['grain'] = 500

    @pytest.mark.skip(reason="Military reaction belongs to the CAMP tier now (D32): the civil district has no army, no war and no military actions in its space. Re-home this test when agents/camp/ exists — do not delete it, the behaviour it asks for is still wanted, just not HERE.")
    def test_heavy_attack_scenario(self):
        """Bot under heavy attack (enemy 3x stronger)."""
        # Overwhelmingly strong enemy
        self.game_state.factions[0].military_strength = 60
        self.game_state.factions[1].military_strength = 10

        action = self.bot.act(self.game_state, game_time=600)

        # Should desperately build military
        assert action.action_type in [
            ActionType.BUILD_BARRACKS,
            ActionType.TRAIN_SOLDIER,
            ActionType.TRAIN_ARCHER,
        ], f"Under heavy attack should build military, got {action.action_type.name}"

    def test_winning_scenario(self):
        """Bot is winning decisively."""
        self.game_state.factions[0].military_strength = 5
        self.game_state.factions[1].military_strength = 20

        action = self.bot.act(self.game_state, game_time=600)

        # Should attack to finish opponent
        assert action.action_type == ActionType.ATTACK, \
            f"When winning should attack, got {action.action_type.name}"

    def test_stalemate_scenario(self):
        """Bot in military stalemate (equal forces)."""
        self.game_state.factions[0].military_strength = 30
        self.game_state.factions[1].military_strength = 30
        self.game_state.factions[1].buildings['barracks'] = 1

        actions = []
        for _ in range(10):
            action = self.bot.act(self.game_state, game_time=800)
            actions.append(action.action_type)

        # Should NOT attack (no advantage)
        attack_count = sum(1 for a in actions if a == ActionType.ATTACK)
        assert attack_count == 0, "Should not attack without advantage"

        # Should build more military or economy
        productive = sum(1 for a in actions if a != ActionType.DO_NOTHING)
        assert productive > 5, "Should continue building during stalemate"

    def test_no_opponent_scenario(self):
        """Bot behavior when there's no opponent."""
        self.game_state.factions = [Faction(id=1)]  # Only bot faction

        action = self.bot.act(self.game_state, game_time=600)

        # Should not crash, should build economy
        assert action is not None
        assert action.action_type != ActionType.ATTACK, \
            "Should not attack when no opponent exists"


class TestEdgeCases:
    """Test NormalBot edge cases and error handling."""

    def setup_method(self):
        """Setup for edge case tests."""
        self.bot = get_opponent('normal', faction_id=1)

    def test_empty_game_state(self):
        """Bot handles empty game state gracefully."""
        game_state = GameState()
        game_state.factions = []

        action = self.bot.act(game_state, game_time=0)

        assert action is not None
        assert action.action_type == ActionType.DO_NOTHING

    def test_wrong_faction_id(self):
        """Bot handles faction not found."""
        game_state = GameState()
        game_state.factions = [Faction(id=0)]  # No faction 1

        action = self.bot.act(game_state, game_time=0)

        assert action is not None
        assert action.action_type == ActionType.DO_NOTHING

    def test_zero_game_time(self):
        """Bot handles start of game (time = 0)."""
        game_state = GameState()
        game_state.factions = [Faction(id=0), Faction(id=1)]
        game_state.factions[1].resources['wood'] = 500
        game_state.factions[1].resources['stone'] = 500

        action = self.bot.act(game_state, game_time=0)

        # Should make a valid decision at game start
        assert action is not None
        assert isinstance(action.action_type, ActionType)

    def test_very_long_game(self):
        """Bot handles very long game times."""
        game_state = GameState()
        game_state.factions = [Faction(id=0), Faction(id=1)]
        game_state.factions[1].resources['wood'] = 1000
        game_state.factions[1].resources['stone'] = 1000

        # Very long game (1000 hours = absurd)
        action = self.bot.act(game_state, game_time=3600000)

        assert action is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
