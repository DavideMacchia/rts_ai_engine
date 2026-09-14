"""
Integration tests for NormalBot behavior over game time.
Tests realistic gameplay scenarios and decision-making.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from simulator.opponents import get_opponent
from simulator.game_state import GameState, Faction
from simulator.actions import ActionType
from simulator.realtime_simulator import RealTimeRTSSimulator


class TestNormalBotBasicBehavior:
    """Test NormalBot makes sensible basic decisions."""

    def setup_method(self):
        """Create a NormalBot instance for each test."""
        self.bot = get_opponent('normal', faction_id=1)

    def test_bot_creation(self):
        """The bot can be created (renamed NormalBot -> TestingBot)."""
        assert self.bot.name == "TestingBot"
        assert self.bot.difficulty == "testing"
        assert self.bot.faction_id == 1

    def test_returns_valid_action(self):
        """Bot always returns a valid action."""
        game_state = GameState()
        game_state.factions = [Faction(id=0), Faction(id=1)]

        action = self.bot.act(game_state, game_time=0)

        assert action is not None
        assert hasattr(action, 'action_type')
        assert isinstance(action.action_type, ActionType)

    def test_builds_economy_when_safe(self):
        """Bot prioritizes economy when safe."""
        game_state = GameState()
        game_state.factions = [Faction(id=0), Faction(id=1)]

        game_state.factions[1].resources['wood'] = 1000
        game_state.factions[1].resources['stone'] = 1000
        game_state.factions[1].resources['grain'] = 500
        game_state.factions[1].population_capacity = 100  # Plenty of housing

        actions = []
        for i in range(10):
            action = self.bot.act(game_state, game_time=i * 60.0)
            actions.append(action.action_type)

        # Should build economic buildings when safe
        economic_actions = [
            ActionType.BUILD_LUMBERYARD,
            ActionType.BUILD_QUARRY,
            ActionType.BUILD_FARM,
            ActionType.BUILD_WELL,
            ActionType.BUILD_MILL,
            ActionType.BUILD_BAKERY,
        ]

        assert any(a in economic_actions for a in actions), \
            f"Expected economic actions, got: {[a.name for a in actions]}"

    def test_builds_housing_when_needed(self):
        """Bot builds houses when population near capacity."""
        game_state = GameState()
        game_state.factions = [Faction(id=0), Faction(id=1)]

        # Give bot resources and make it need housing
        game_state.factions[1].resources['wood'] = 1000
        game_state.factions[1].resources['stone'] = 1000
        game_state.factions[1].population = 45
        game_state.factions[1].population_capacity = 50  # Near capacity!

        action = self.bot.act(game_state, game_time=0)

        assert action.action_type == ActionType.BUILD_HOUSE, \
            f"Expected BUILD_HOUSE when near capacity, got {action.action_type.name}"


class TestNormalBotDefensiveBehavior:
    """Test NormalBot defensive reactions."""

    def setup_method(self):
        """Create a NormalBot instance for each test."""
        self.bot = get_opponent('normal', faction_id=1)

    @pytest.mark.skip(reason="Military reaction belongs to the CAMP tier now (D32): the civil district has no army, no war and no military actions in its space. Re-home this test when agents/camp/ exists — do not delete it, the behaviour it asks for is still wanted, just not HERE.")
    def test_defends_when_under_attack(self):
        """Bot builds military when under heavy attack."""
        game_state = GameState()
        game_state.factions = [Faction(id=0), Faction(id=1)]

        game_state.factions[0].military_strength = 50
        game_state.factions[1].military_strength = 10

        game_state.factions[1].resources['wood'] = 1000
        game_state.factions[1].resources['stone'] = 1000
        game_state.factions[1].resources['grain'] = 500

        action = self.bot.act(game_state, game_time=0)

        # Should prioritize military defense
        defensive_actions = [
            ActionType.BUILD_BARRACKS,
            ActionType.TRAIN_SOLDIER,
            ActionType.TRAIN_ARCHER,
        ]

        assert action.action_type in defensive_actions, \
            f"Expected defensive action when under attack, got {action.action_type.name}"

    @pytest.mark.skip(reason="Military reaction belongs to the CAMP tier now (D32): the civil district has no army, no war and no military actions in its space. Re-home this test when agents/camp/ exists — do not delete it, the behaviour it asks for is still wanted, just not HERE.")
    def test_builds_barracks_first_when_threatened(self):
        """Bot builds barracks when threatened and has no military infrastructure."""
        game_state = GameState()
        game_state.factions = [Faction(id=0), Faction(id=1)]

        game_state.factions[0].military_strength = 40
        game_state.factions[1].military_strength = 5
        game_state.factions[1].buildings['barracks'] = 0  # No barracks

        game_state.factions[1].resources['wood'] = 1000
        game_state.factions[1].resources['stone'] = 1000
        game_state.factions[1].resources['grain'] = 500

        action = self.bot.act(game_state, game_time=0)

        assert action.action_type == ActionType.BUILD_BARRACKS, \
            f"Expected BUILD_BARRACKS when threatened with no barracks, got {action.action_type.name}"

    def test_trains_units_when_has_barracks_and_threatened(self):
        """Bot trains units when it has barracks and is threatened."""
        game_state = GameState()
        game_state.factions = [Faction(id=0), Faction(id=1)]

        game_state.factions[0].military_strength = 40
        game_state.factions[1].military_strength = 5
        game_state.factions[1].buildings['barracks'] = 1  # Has barracks

        game_state.factions[1].resources['wood'] = 1000
        game_state.factions[1].resources['stone'] = 1000
        game_state.factions[1].resources['grain'] = 500

        action = self.bot.act(game_state, game_time=0)

        assert action.action_type in [ActionType.TRAIN_SOLDIER, ActionType.TRAIN_ARCHER], \
            f"Expected TRAIN unit when has barracks and threatened, got {action.action_type.name}"


class TestNormalBotOffensiveBehavior:
    """Test NormalBot offensive/attack behavior."""

    def setup_method(self):
        """Create a NormalBot instance for each test."""
        self.bot = get_opponent('normal', faction_id=1)

    def test_attacks_when_strong(self):
        """Bot attacks when it has clear military advantage."""
        game_state = GameState()
        game_state.factions = [Faction(id=0), Faction(id=1)]

        # Bot is much stronger (1.5x+ advantage)
        game_state.factions[0].military_strength = 10
        game_state.factions[1].military_strength = 30  # 3x stronger

        action = self.bot.act(game_state, game_time=0)

        assert action.action_type == ActionType.ATTACK, \
            f"Expected ATTACK when having advantage, got {action.action_type.name}"
        assert action.target_faction_id == 0, \
            "Attack should target opponent (faction 0)"

    def test_does_not_attack_when_weak(self):
        """Bot doesn't attack when military is weak."""
        game_state = GameState()
        game_state.factions = [Faction(id=0), Faction(id=1)]

        game_state.factions[0].military_strength = 30
        game_state.factions[1].military_strength = 10

        # Give resources to build instead
        game_state.factions[1].resources['wood'] = 1000
        game_state.factions[1].resources['stone'] = 1000
        game_state.factions[1].resources['grain'] = 500

        action = self.bot.act(game_state, game_time=0)

        assert action.action_type != ActionType.ATTACK, \
            "Bot should not attack when weaker"

    def test_does_not_attack_when_equal(self):
        """Bot doesn't attack when military is roughly equal."""
        game_state = GameState()
        game_state.factions = [Faction(id=0), Faction(id=1)]

        # Equal strength (not enough advantage)
        game_state.factions[0].military_strength = 20
        game_state.factions[1].military_strength = 22  # Only 1.1x

        game_state.factions[1].resources['wood'] = 1000
        game_state.factions[1].resources['stone'] = 1000
        game_state.factions[1].resources['grain'] = 500

        action = self.bot.act(game_state, game_time=0)

        assert action.action_type != ActionType.ATTACK, \
            "Bot should not attack without clear advantage (needs 1.5x)"


class TestNormalBotGameProgression:
    """Test NormalBot behavior over full game progression."""

    def setup_method(self):
        """Create simulator with NormalBot."""
        self.sim = RealTimeRTSSimulator(num_factions=2, max_game_time=3600.0)
        self.bot = get_opponent('normal', faction_id=1)

    def test_survives_full_game(self):
        """Bot can play through a full game without errors."""
        steps = 0
        max_steps = 100

        while not self.sim.is_done() and steps < max_steps:
            bot_action = self.bot.act(self.sim.state, self.sim.game_time)

            from simulator.actions import Action
            opponent_action = Action(faction_id=0, action_type=ActionType.DO_NOTHING)

            self.sim.step({0: opponent_action, 1: bot_action}, delta_time=60.0)
            steps += 1

        # Bot should have built things
        bot_faction = self.sim.state.get_faction(1)
        total_buildings = sum(bot_faction.buildings.values())

        assert total_buildings > 0, "Bot should have built buildings during game"

    def test_economy_grows_over_time(self):
        """Bot's economy should grow over time when safe."""
        steps_to_run = 30
        building_counts = []

        for step in range(steps_to_run):
            bot_faction = self.sim.state.get_faction(1)
            building_counts.append(sum(bot_faction.buildings.values()))

            bot_action = self.bot.act(self.sim.state, self.sim.game_time)

            from simulator.actions import Action
            opponent_action = Action(faction_id=0, action_type=ActionType.DO_NOTHING)

            self.sim.step({0: opponent_action, 1: bot_action}, delta_time=60.0)

        # Economy should generally trend upward (allowing for some variance)
        early_buildings = sum(building_counts[:10]) / 10
        late_buildings = sum(building_counts[-10:]) / 10

        assert late_buildings >= early_buildings, \
            f"Economy should grow over time: early={early_buildings:.1f}, late={late_buildings:.1f}"

    def test_builds_military_eventually(self):
        """Bot should build military infrastructure eventually (not pure eco)."""
        steps = 0
        max_steps = 60

        while steps < max_steps:
            bot_action = self.bot.act(self.sim.state, self.sim.game_time)

            from simulator.actions import Action
            opponent_action = Action(faction_id=0, action_type=ActionType.DO_NOTHING)

            self.sim.step({0: opponent_action, 1: bot_action}, delta_time=60.0)
            steps += 1

        # Check if bot has military infrastructure or units
        bot_faction = self.sim.state.get_faction(1)
        has_military = (
            bot_faction.buildings.get('barracks', 0) > 0 or
            bot_faction.military_strength > 0
        )

        assert has_military, \
            "Bot should build military infrastructure/units eventually (not pure economy)"


class TestNormalBotResourceManagement:
    """Test NormalBot resource management behavior."""

    def setup_method(self):
        """Create a NormalBot instance for each test."""
        self.bot = get_opponent('normal', faction_id=1)

    def test_does_nothing_when_no_resources(self):
        """Bot does nothing when it can't afford anything."""
        game_state = GameState()
        game_state.factions = [Faction(id=0), Faction(id=1)]

        game_state.factions[1].resources['wood'] = 0
        game_state.factions[1].resources['stone'] = 0
        game_state.factions[1].resources['grain'] = 0

        action = self.bot.act(game_state, game_time=0)

        assert action.action_type == ActionType.DO_NOTHING, \
            f"Expected DO_NOTHING when broke, got {action.action_type.name}"

    def test_prioritizes_affordable_actions(self):
        """Bot chooses actions it can actually afford."""
        game_state = GameState()
        game_state.factions = [Faction(id=0), Faction(id=1)]

        # Limited resources - can only afford cheap actions
        game_state.factions[1].resources['wood'] = 40
        game_state.factions[1].resources['stone'] = 40
        game_state.factions[1].resources['grain'] = 10
        game_state.factions[1].population_capacity = 100  # Not housing limited

        action = self.bot.act(game_state, game_time=0)

        # Should either do nothing or build something cheap (well, farm)
        # Should NOT try to build expensive things like barracks
        assert action.action_type != ActionType.BUILD_BARRACKS, \
            "Bot shouldn't try expensive actions when low on resources"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
