"""
Real-Time RTS Environment for RL Training
Uses JSON configuration for all parameters including normalization constants.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.realtime_simulator import RealTimeRTSSimulator
from simulator.actions import Action, ActionType, index_to_action
from simulator.config import BUILDING_COSTS, UNIT_TRAINING_COST, EDIBLE_FOODS, DISTRICT_RADIUS
from simulator.map import TileType, DepositType
from simulator.opponents import get_opponent
from simulator.opponents.base import ScriptedOpponent
from simulator.policies import POLICY_ACTIONS, PopulationRate, policy_action_is_available
from config.config_loader import load_env_config

#: The district's plot, as the agent sees it: a (2R+1)² patch centred on its warehouse.
PLOT_SIDE = 2 * DISTRICT_RADIUS + 1
PLOT_TILES = PLOT_SIDE * PLOT_SIDE

#: Per-tile channels: terrain one-hot, deposit one-hot, richness, occupied, distance.
PATCH_CHANNELS = len(TileType) + len(DepositType) + 3

#: The only thing the district cannot do here: ATTACK (there is no enemy on a solo plot — it
#: develops its own territory). Everything else military — barracks, training, arming, the
#: conscription that turns surplus population into soldiers — it DOES (district-unified design,
#: overrides D32). The army it raises is part of what its territory is WORTH, scored below.
COMBAT_ACTIONS = {'ATTACK'}
MILITARY_ACTIONS = COMBAT_ACTIONS      # kept name for callers; only ATTACK is withheld now

#: The GOODS a district can make, and their GRADE: how deep in the production chain each one
#: sits. Higher grade is worth more, so the score rewards climbing the chain — turning raw
#: land into finished goods and military kit — not hoarding a heap of one raw commodity.
RESOURCE_GRADE = {
    # grade 1: raw, taken straight from the land (exploiting the territory)
    'wood': 1, 'stone': 1, 'clay': 1, 'iron_ore': 1, 'grain': 1, 'water': 1,
    'vegetable': 1, 'fruit': 1, 'meat': 1,
    # grade 2: one processing step
    'wood_logs': 2, 'flour': 2, 'clay_bricks': 2, 'stone_bricks': 2,
    'iron_ingots': 2, 'leather': 2, 'wool': 2, 'wooden_weapon': 2,
    # grade 3: finished / high-level, incl. the iron military kit
    'bread': 3, 'clothes': 3, 'iron_weapon': 3, 'armour': 3,
}

#: The worth of one unit of a good at each grade. GEOMETRIC, not linear: a raw commodity is
#: plentiful and cheap, a finished good is scarce and dear, so refining a grade up is always a
#: net gain even though it consumes several raw units — which is exactly the incentive to climb
#: the chain rather than hoard raw land. (Raw is voluminous, so a gentle per-unit value keeps it
#: from swamping the finished goods by sheer quantity.)
GRADE_VALUE = {1: 0.25, 2: 3.0, 3: 12.0}

#: Kept for the observation and older callers: the distinct goods a developed district makes.
GOODS = [
    'bread', 'meat', 'vegetable', 'fruit',
    'clay_bricks', 'stone_bricks', 'wood_logs',
    'clothes',
    'iron_ingots', 'wooden_weapon', 'iron_weapon', 'armour',
]

# --- the graded valuation: what a territory is WORTH (district-unified design) ---
# The district is scored on a SUM of the total value of everything it has made, with higher
# grade worth more. Three ledgers — people (by skill/tier), the army (by kit), and goods (by
# chain depth) — so the agent is paid to exploit its land, climb to high-grade production, and
# turn its surplus population into soldiers. No population cap or target: the limit is the plot.
PERSON_BASE_VALUE = 2.0        # a living adult, before any skill
KID_VALUE = 1.0               # a child: a worker in waiting, worth less than a grown one
SKILL_TIER_VALUE = 1.0        # a master adds (SKILL_TIER_VALUE x tier): an L3 smith ~+3
SOLDIER_BASE_VALUE = 6.0      # a trained soldier, times its kit's strength (man-at-arms ~1.8x)
GOOD_UNIT_VALUE = 0.08        # per unit of a good, times its grade


class RealTimeRTSEnv(gym.Env):
    """Real-time RTS environment for RL training."""

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        decision_interval: Optional[float] = None,
        config_path: Optional[str] = None,
        env_config_path: Optional[str] = None,
        opponent_type: str = 'balanced',
        spatial: bool = False,
        objective: str = 'conquest',
    ):
        super().__init__()

        # WHAT THIS BRAIN IS FOR (D32).
        #
        # 'economy' — the civil district: it makes PEOPLE and GOODS. There is no enemy, no
        #   army and no conquest, and the military actions are not in its action space at all.
        #   War is the military camp's job, and the camp is a different model. Pay one brain
        #   for a conquest and it will build barracks first and develop never — you cannot ask
        #   it to win a war and raise a civilisation and then be surprised that it does the
        #   first.
        #
        # 'conquest' — the older objective, kept so every prior checkpoint and measurement can
        #   still be reproduced.
        self.objective = objective

        # `spatial` is the whole of D28. Off, the agent names a building type and the WORLD
        # picks the ground (D27) — the flat baseline every previous checkpoint was trained
        # on, kept alive so the spatial agent has something honest to be compared against.
        # On, the agent sees its plot and says WHERE.
        self.spatial = spatial

        # Load environment configuration
        self.env_config = load_env_config(env_config_path)
        env_cfg = self.env_config['environment']
        obs_cfg = self.env_config['observation_space']
        sim_cfg = self.env_config['simulator']

        # Normalization constants from config
        norm_cfg = self.env_config.get('normalization', {})
        self.MAX_RESOURCE = norm_cfg.get('max_resource', 500.0)
        self.MAX_POPULATION = norm_cfg.get('max_population', 50.0)
        self.MAX_BUILDINGS = norm_cfg.get('max_buildings', 10.0)
        self.MAX_MILITARY = norm_cfg.get('max_military', 50.0)
        self.MAX_UNITS = norm_cfg.get('max_units', 50.0)
        self.MAX_BARRACKS = norm_cfg.get('max_barracks', 5.0)
        self.MAX_WAREHOUSE = norm_cfg.get('max_warehouse', 5.0)
        self.MAX_PROCESSING = norm_cfg.get('max_processing', 5.0)

        # Decision interval (allow override)
        self.decision_interval = decision_interval or env_cfg['decision_interval']
        self.max_game_time = env_cfg['max_game_time']

        # Setup simulator. The civil district plays alone: an opponent it cannot fight and
        # cannot be conquered by is not an opponent, it is noise in the reward.
        self.num_factions = 1 if self.objective == 'economy' else sim_cfg['num_factions']
        self.sim = RealTimeRTSSimulator(
            num_factions=self.num_factions,
            max_game_time=self.max_game_time
        )
        self.agent_faction_id = sim_cfg['agent_faction_id']

        # Setup opponent AI
        opponent_id = 1
        self.opponent_type = opponent_type
        self.opponent: Optional[ScriptedOpponent] = (
            None if self.objective == 'economy'
            else get_opponent(opponent_type, faction_id=opponent_id)
        )

        self.last_decision_time = 0.0
        self.action_history = []

        # Tracking for logging
        self.episode_wins = 0
        self.episode_losses = 0
        self.episode_reward_components = {}

        # Feature keys from config
        self.feature_keys = env_cfg['feature_keys']
        observation_size = env_cfg['observation_space_size']

        if len(self.feature_keys) != observation_size:
            raise ValueError(
                f"Feature keys count ({len(self.feature_keys)}) doesn't match "
                f"observation_space_size ({observation_size})"
            )

        # Gymnasium spaces
        scalar_space = spaces.Box(
            low=obs_cfg['low'],
            high=obs_cfg['high'],
            shape=(observation_size,),
            dtype=getattr(np, obs_cfg['dtype'])
        )
        self.action_space_size = len(list(ActionType))

        if self.spatial:
            # The plot is a GRID, so it is given to the agent as one: a stack of feature
            # maps over a 13x13 patch, next to the flat district summary. (A grid is what a
            # convolution is for. The Transformer kept in `extractors.py` is the right tool
            # for a variable-size SET of entities — which this is not.)
            self.observation_space = spaces.Dict({
                'scalars': scalar_space,
                'patch': spaces.Box(low=0.0, high=1.0,
                                    shape=(PATCH_CHANNELS, PLOT_SIDE, PLOT_SIDE),
                                    dtype=np.float32),
            })
            # FACTORED, not flattened: one head over WHAT to build, one over WHERE.
            # Flattening to Discrete(n_actions x 169) would multiply the action space out to
            # thousands of entries that share no structure — every (type, tile) pair would
            # have to be learned on its own, and nothing learned about a tile would transfer
            # to the next building put on it.
            self.action_space = spaces.MultiDiscrete([self.action_space_size, PLOT_TILES])
        else:
            self.observation_space = scalar_space
            self.action_space = spaces.Discrete(self.action_space_size)

    def action_masks(self) -> np.ndarray:
        """Return boolean mask of valid actions."""
        agent = self.sim.state.factions[self.agent_faction_id]
        mask = np.ones(self.action_space_size, dtype=bool)

        # One entry per BUILD_* member of ActionType. Anything else is unreachable:
        # the loop below only ever looks up names that ActionType actually defines.
        # Placing a warehouse founds a district, so it belongs to the macro tier
        # (FOUND_DISTRICT(region)), not here.
        building_map = {
            'BUILD_LUMBERYARD': 'lumberyard',
            'BUILD_QUARRY': 'quarry',
            'BUILD_CLAY_PIT': 'clay_pit',
            'BUILD_MINE': 'mine',
            'BUILD_WELL': 'well',
            'BUILD_FARM': 'farm',
            'BUILD_MILL': 'mill',
            'BUILD_BAKERY': 'bakery',
            'BUILD_HUNTING_SHED': 'hunting_shed',
            'BUILD_CATTLE_SHED': 'cattle_shed',
            'BUILD_VEGETABLE_GARDEN': 'vegetable_garden',
            'BUILD_ORCHARD': 'orchard',
            'BUILD_SCHOOL': 'school',
            'BUILD_HOUSE': 'house',
            'BUILD_POTTERY': 'pottery',
            'BUILD_CARPENTRY': 'carpentry',
            'BUILD_STONE_CUTTER': 'stone_cutter',
            'BUILD_BLACKSMITH': 'blacksmith',
            'BUILD_SMELTER': 'smelter',
            'BUILD_TAILOR': 'tailor',
            'BUILD_DORMITORY': 'dormitory',
            'BUILD_WATCHTOWER': 'watchtower',
            'BUILD_DEFENSIVE_WALL_WOOD': 'defensive_wall_wood',
            'BUILD_DEFENSIVE_WALL_STONE': 'defensive_wall_stone',
            'BUILD_BARRACKS': 'barracks',
        }

        unit_map = {
            'TRAIN_SOLDIER': 'soldier',
            'TRAIN_MAN_AT_ARMS': 'man_at_arms',
            'TRAIN_ARCHER': 'archer'
        }

        for i in range(self.action_space_size):
            try:
                action_type = index_to_action(i)
            except (IndexError, ValueError):
                mask[i] = False
                continue

            action_name = action_type.name

            # On a solo plot there is no enemy, so ATTACK is withheld — but the district DOES
            # build barracks, train and arm soldiers, and raise the military supply chain
            # (district-unified design). Only combat itself is out of scope here.
            if self.objective == 'economy' and action_name in COMBAT_ACTIONS:
                mask[i] = False
                continue

            if action_name in building_map:
                building_type = building_map[action_name]
                if building_type in BUILDING_COSTS:
                    mask[i] = agent.can_afford(BUILDING_COSTS[building_type])
                else:
                    mask[i] = False

            elif action_name in unit_map:
                unit_type = unit_map[action_name]
                has_barracks = agent.get_building_count('barracks') > 0
                can_afford = agent.can_afford(UNIT_TRAINING_COST.get(unit_type, {}))
                mask[i] = has_barracks and can_afford

            elif action_name == 'ATTACK':
                has_military = (
                    agent.get_unit_count('soldier') > 0 or
                    agent.get_unit_count('archer') > 0
                )
                # An army already marching cannot be re-committed: the order is a no-op
                # that still costs the decision. Mask it, so the agent spends the march
                # building the next army instead of shouting at an empty road.
                marching = self.sim.state.is_marching(self.agent_faction_id)
                mask[i] = has_military and not marching

            elif action_type in POLICY_ACTIONS:
                # Hide a policy action that is already in force: re-asserting it would
                # burn a decision and teach the agent nothing.
                mask[i] = policy_action_is_available(action_type, agent.capital.policies)

            elif action_name == 'DO_NOTHING':
                mask[i] = True

            else:
                mask[i] = False

        if not self.spatial:
            return mask

        # SB3 wants one flat vector holding each head's mask, back to back.
        return np.concatenate([mask, self.tile_masks()])

    # --- what this brain is paid for: the graded value of the territory ----------

    def _human_value(self, district) -> float:
        """The district's PEOPLE, graded by expertise. A living adult is worth
        PERSON_BASE_VALUE; a master of a high trade is worth much more (an L3 smith adds ~3);
        a child is a worker in waiting. Skill and rank are what make a person valuable, so the
        agent is paid to raise experts, not just mouths."""
        from simulator.professions import PROFESSION_TIER
        total = 0.0
        for sim in district.sims:
            if sim.is_kid:
                total += KID_VALUE
                continue
            best = 0.0
            for prof, skill in sim.skills.items():
                best = max(best, skill * PROFESSION_TIER.get(prof, 1))
            total += PERSON_BASE_VALUE + SKILL_TIER_VALUE * best
        return total

    def _army_value(self, faction) -> float:
        """The ARMY, graded by kit: each soldier is worth SOLDIER_BASE_VALUE times its
        strength, so a man-at-arms in iron is worth ~1.8 of a base spearman. This is where the
        surplus population — trained and armed — turns into score."""
        from simulator.config import UNIT_STRENGTH
        return SOLDIER_BASE_VALUE * sum(
            n * UNIT_STRENGTH.get(u, 1.0) for u, n in faction.units.items())

    def _goods_value(self, faction) -> float:
        """The GOODS on hand, each unit graded by its depth in the production chain: raw land
        counts a little, a finished loaf or an iron sword counts triple. Rewards exploiting the
        territory AND refining what it yields into high-grade product."""
        return GOOD_UNIT_VALUE * sum(
            faction.get_resource(g) * GRADE_VALUE[grade] for g, grade in RESOURCE_GRADE.items())

    def _progress(self):
        """The three ledgers whose graded sum is the territory's worth: people, army, goods."""
        faction = self.sim.state.factions[self.agent_faction_id]
        district = faction.capital
        return np.array([
            self._human_value(district),
            self._army_value(faction),
            self._goods_value(faction),
        ], dtype=np.float64)

    def score(self) -> float:
        """What the territory is WORTH: the graded sum of everything it has made — its people
        (by expertise), its army (by kit), and its goods (by chain depth). Higher grade is
        worth more, so the score rewards exploiting the land and climbing to high-grade
        production and a real army, with no population cap or target (the plot is the limit).

        The step reward is the DELTA of exactly this, so shaping and objective never disagree,
        and an idle district is paid nothing for what it already had (D2)."""
        return float(self._progress().sum())

    def _economy_reward(self) -> float:
        """Pay for the CHANGE in the territory's worth, never for the state (D2): another
        expert raised, another soldier armed, another good refined a grade higher. Losses are
        paid for too, with the same coin."""
        progress = self._progress()
        delta = progress - self._prev_progress
        self._prev_progress = progress
        return float(delta.sum())

    def _get_opponent_action(self) -> Action:
        """Get action from scripted opponent."""
        return self.opponent.act(self.sim.state, self.sim.game_time)

    def _get_observation(self) -> np.ndarray:
        """Get current observation using normalization constants from config."""
        state = self.sim.state
        agent = state.factions[self.agent_faction_id]
        opponent = state.factions[1] if len(state.factions) > 1 else None
        policies = agent.capital.policies

        obs_dict = {
            'time_ratio': min(self.sim.game_time / self.max_game_time, 1.0),
            'wood_ratio': min(agent.get_resource('wood') / self.MAX_RESOURCE, 1.0),
            'stone_ratio': min(agent.get_resource('stone') / self.MAX_RESOURCE, 1.0),
            'clay_ratio': min(agent.get_resource('clay') / self.MAX_RESOURCE, 1.0),
            'grain_ratio': min(agent.get_resource('grain') / self.MAX_RESOURCE, 1.0),
            'water_ratio': min(agent.get_resource('water') / self.MAX_RESOURCE, 1.0),
            # Edible food (bread/meat/vegetable/fruit), NOT grain — grain is an intermediate
            # and nobody eats it (D16).
            'food_ratio': min(agent.get_total_food() / self.MAX_RESOURCE, 1.0),
            # How many distinct foods are in stock, normalised — the state the variety
            # happiness bonus reads, so the agent can see why diversifying pays.
            'food_variety_ratio': agent.get_food_variety() / max(1, len(EDIBLE_FOODS)),
            # Human capital: mean worker skill (0-1). Lets the agent see whether its
            # workforce is skilled, and thus whether a school is worth building.
            'skill_ratio': agent.capital.get_mean_skill(),
            'population_ratio': min(agent.population / self.MAX_POPULATION, 1.0),
            'population_capacity_ratio': min(agent.population_capacity / self.MAX_POPULATION, 1.0),
            'farm_count': min(agent.get_building_count('farm') / self.MAX_BUILDINGS, 1.0),
            'lumberyard_count': min(agent.get_building_count('lumberyard') / self.MAX_BUILDINGS, 1.0),
            'quarry_count': min(agent.get_building_count('quarry') / self.MAX_BUILDINGS, 1.0),
            'house_count': min(agent.get_building_count('house') / self.MAX_BUILDINGS, 1.0),
            'barracks_count': min(agent.get_building_count('barracks') / self.MAX_BARRACKS, 1.0),
            'warehouse_count': min(agent.get_building_count('warehouse') / self.MAX_WAREHOUSE, 1.0),
            'mill_count': min(agent.get_building_count('mill') / self.MAX_PROCESSING, 1.0),
            'bakery_count': min(agent.get_building_count('bakery') / self.MAX_PROCESSING, 1.0),
            'soldier_count': min(agent.get_unit_count('soldier') / self.MAX_UNITS, 1.0),
            'military_strength': min(agent.military_strength / self.MAX_MILITARY, 1.0),
            # Is our army on the road? An attack MARCHES (D23), and an army in transit
            # cannot be sent again. Without this bit the agent cannot tell "attack now"
            # from "I already did, it is still walking": the ATTACK trigger is then not a
            # function of the observation at all, and no clone of it can learn it (D25).
            'army_marching': 1.0 if state.is_marching(self.agent_faction_id) else 0.0,
            'opponent_military_strength': min(opponent.military_strength / self.MAX_MILITARY, 1.0) if opponent else 0.0,
            'opponent_warehouse_count': min(opponent.get_building_count('warehouse') / self.MAX_WAREHOUSE, 1.0) if opponent else 0.0,
            'opponent_population': min(opponent.population / self.MAX_POPULATION, 1.0) if opponent else 0.0,

            # Age structure: `population_ratio` alone hides that kids do not work.
            'kids_ratio': min(agent.kids / self.MAX_POPULATION, 1.0),
            'adults_ratio': min(agent.adults / self.MAX_POPULATION, 1.0),
            'elders_ratio': min(agent.elders / self.MAX_POPULATION, 1.0),

            # The state the policies act on, and the policies themselves. Without
            # these the agent would be steering a variable it cannot see.
            'happiness_ratio': agent.happiness / 100.0,
            'policy_minimum_food': 1.0 if policies.minimum_food else 0.0,
            'policy_forced_conscription': 1.0 if policies.forced_conscription else 0.0,
            'policy_population_rate': {
                PopulationRate.DECREASE: 0.0,
                PopulationRate.MAINTAIN: 0.5,
                PopulationRate.INCREASE: 1.0,
            }[policies.population_rate],
        }

        observation = np.array(
            [obs_dict[key] for key in self.feature_keys],
            dtype=np.float32
        )

        if not self.spatial:
            return observation

        return {'scalars': observation, 'patch': self._get_patch()}

    # --- the plot, as the agent sees it (D28) ---------------------------------

    def plot_origin(self):
        """Top-left map coordinate of the district's plot."""
        cx, cy = self.sim.state.factions[self.agent_faction_id].capital.center
        return cx - DISTRICT_RADIUS, cy - DISTRICT_RADIUS

    def tile_of(self, index: int):
        """Map coordinate of a tile head's choice."""
        ox, oy = self.plot_origin()
        return ox + index // PLOT_SIDE, oy + index % PLOT_SIDE

    def _get_patch(self) -> np.ndarray:
        """One feature map per property of the ground, over the district's own land.

        Everything the placement rules read has to be in here, or the agent is being asked
        to choose a tile on evidence it does not have — the same mistake that made ATTACK
        unlearnable until `army_marching` was added to the observation (D25).
        """
        patch = np.zeros((PATCH_CHANNELS, PLOT_SIDE, PLOT_SIDE), dtype=np.float32)
        game_map = self.sim.state.map
        if game_map is None:
            return patch

        terrains = list(TileType)
        deposits = list(DepositType)
        ox, oy = self.plot_origin()

        for i in range(PLOT_SIDE):
            for j in range(PLOT_SIDE):
                tile = game_map.at(ox + i, oy + j)
                if tile is None:
                    continue        # off the map: every channel stays 0
                patch[terrains.index(tile.terrain), i, j] = 1.0
                if tile.deposit is not None:
                    patch[len(terrains) + deposits.index(tile.deposit), i, j] = 1.0
                    patch[len(terrains) + len(deposits), i, j] = tile.richness
                patch[len(terrains) + len(deposits) + 1, i, j] = 0.0 if tile.building is None else 1.0
                patch[len(terrains) + len(deposits) + 2, i, j] = (
                    max(abs(i - DISTRICT_RADIUS), abs(j - DISTRICT_RADIUS)) / DISTRICT_RADIUS
                )
        return patch

    def tile_masks(self) -> np.ndarray:
        """Which tiles can carry a building at all: a tile where at least the SMALLEST footprint
        (2x2) fits — on the map, inside the plot, and free.

        It is deliberately generic — it does NOT know which building is being placed, because
        the two heads are masked independently. So "a mine needs an iron vein", or "a 3x3 needs
        more room than this corner has", is not something the mask can express: naming a tile
        that cannot take THIS building is a legal move with a bad outcome (the build is refused
        and the decision wasted), and the agent learns the geometry rather than being shielded
        from it. What the mask does guarantee is that a named tile can carry SOMETHING.
        """
        mask = np.zeros(PLOT_TILES, dtype=bool)
        game_map = self.sim.state.map
        if game_map is None:
            return np.ones(PLOT_TILES, dtype=bool)

        for index in range(PLOT_TILES):
            x, y = self.tile_of(index)
            # a 2x2 lot fits here (the minimum any building needs)
            cells = [(x + dx, y + dy) for dx in (0, 1) for dy in (0, 1)]
            mask[index] = all(
                game_map.at(cx, cy) is not None and game_map.at(cx, cy).is_buildable
                for cx, cy in cells
            )

        if not mask.any():
            mask[:] = True      # a plot with no ground left: never hand SB3 an empty mask
        return mask

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # The simulator draws from the GLOBAL numpy RNG (births, starvation, combat
        # building damage), which `super().reset(seed)` does not touch — it only seeds
        # `self.np_random`. Without this line an episode is not reproducible from its
        # seed, and even a deterministic policy scores differently run to run. It went
        # unnoticed while births were so rare that the simulation was near-deterministic.
        if seed is not None:
            np.random.seed(seed)

        self.sim = RealTimeRTSSimulator(
            num_factions=self.num_factions,
            max_game_time=self.max_game_time
        )
        self.last_decision_time = 0.0
        self.action_history = []

        # Reset opponent AI
        if self.objective == 'economy':
            self.opponent = None
        else:
            self.opponent = get_opponent(self.opponent_type, faction_id=1)
        self._prev_progress = self._progress()

        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action):
        """Execute action and simulate until next decision point.

        Spatial: `action` is (what, where). The tile is only read for a BUILD — for every
        other action the tile head is noise, and it is ignored rather than being given a
        meaning it does not have.
        """
        tile = None
        if self.spatial:
            action, tile_index = int(action[0]), int(action[1])
            if index_to_action(action).name.startswith('BUILD_'):
                tile = self.tile_of(tile_index)

        action_type = index_to_action(action)

        action_obj = Action(
            action_type=action_type,
            faction_id=self.agent_faction_id,
            target_faction_id=1 if action_type == ActionType.ATTACK else None,
            target_tile=tile,
        )

        self.action_history.append(action_type)

        actions_dict = {self.agent_faction_id: action_obj}
        if self.opponent is not None:
            actions_dict[1] = self._get_opponent_action()
        state, rewards, terminated = self.sim.step(actions_dict, delta_time=self.decision_interval)

        self.last_decision_time = self.sim.game_time

        observation = self._get_observation()

        # Reward from simulator (already calculated by RewardCalculator)
        reward = rewards.get(self.agent_faction_id, 0.0)
        if self.objective == 'economy':
            reward = self._economy_reward()

        # `terminated`: someone lost their last warehouse. A real outcome, value 0 after.
        # `truncated`:  the clock ran out. NOT an outcome — SB3 bootstraps the value
        #               function across it, so the agent is not taught that the world
        #               ends at the horizon, and is paid nothing for stalling.
        truncated = (not terminated) and self.sim.is_out_of_time()

        info = {
            'winner': state.winner,          # None on truncation: nobody won
            'time': self.sim.game_time,
            'action': action_type.name,
            'terminated': terminated,
            'truncated': truncated,
            'agent_population': state.factions[self.agent_faction_id].population,
            'agent_military': state.factions[self.agent_faction_id].military_strength,
            'agent_buildings': sum(state.factions[self.agent_faction_id].buildings.values()),
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
