"""Debug test to see what actions the bot is taking."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.realtime_simulator import RealTimeRTSSimulator
from simulator.opponents import NormalBot  # deprecated alias of TestingBot
from simulator.actions import ActionType


def test_bot_actions():
    """Run bot for a while and print all actions."""
    sim = RealTimeRTSSimulator(num_factions=1)
    bot = NormalBot(faction_id=0)

    duration_seconds = 12 * 3600  # 12 hours
    time_step = 60.0

    print("Bot Actions Log:")
    print("="*60)

    game_time = 0.0
    action_counts = {}

    while game_time < duration_seconds:
        action = bot.act(sim.state, game_time)

        # Count actions
        action_type_str = str(action.action_type)
        action_counts[action_type_str] = action_counts.get(action_type_str, 0) + 1

        # Print non-DO_NOTHING actions
        if action.action_type != ActionType.DO_NOTHING:
            hour = game_time / 3600
            faction = sim.state.factions[0]
            print(f"[{hour:5.1f}h] {action.action_type.name:<20} "
                  f"(pop={faction.population}/{faction.population_capacity}, "
                  f"wood={faction.resources.get('wood', 0):.0f}, "
                  f"stone={faction.resources.get('stone', 0):.0f})")

        sim.step({0: action}, delta_time=time_step)
        game_time += time_step

    # Summary
    print("\n" + "="*60)
    print("Action Summary:")
    for action_type, count in sorted(action_counts.items()):
        print(f"  {action_type}: {count}")

    # Final state
    faction = sim.state.factions[0]
    print(f"\nFinal State:")
    print(f"  Population: {faction.population}/{faction.population_capacity}")
    print(f"  Buildings: {sum(faction.buildings.values())}")
    for building_type, count in sorted(faction.buildings.items()):
        if count > 0:
            print(f"    {building_type}: {count}")


if __name__ == '__main__':
    test_bot_actions()
