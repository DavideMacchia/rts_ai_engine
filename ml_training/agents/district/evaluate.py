"""Measure the oracle: CONQUESTS vs easy / medium / hard, plus the population floor.

The metric is conquest. A timeout is not a win — it is the absence of a result (D14) —
so it is reported separately and never counted as one. The policy is always evaluated
DETERMINISTICALLY: a stochastic policy's win rate flatters it, and we have been burned
by that before.

`pop_min` is the smallest population the faction ever fell to, averaged over episodes.
It is the guard-rail on the rusher: recruiting eats adults, and a bot that converts its
whole civilian population into soldiers wins the battle and then starves (see D22).

    python -m agents.district.evaluate --model checkpoints/district/district_agent_bc_v14.zip
    python -m agents.district.evaluate --expert hard      # score the scripted bot instead
"""

import os
import sys
import argparse
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.district.env import RealTimeRTSEnv
from simulator.opponents import get_opponent
from simulator.actions import action_to_index


def evaluate(policy_fn, opponent: str, n_episodes: int, seed0: int = 0, spatial: bool = False):
    env = RealTimeRTSEnv(opponent_type=opponent, spatial=spatial)
    outcomes = Counter()
    pop_mins, armies = [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        faction = env.sim.state.factions[0]
        done, pop_min, army_max = False, float('inf'), 0.0
        while not done:
            mask = env.action_masks()
            action = policy_fn(obs, mask, env)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            pop_min = min(pop_min, faction.population)
            army_max = max(army_max, faction.military_strength)

        winner = info.get('winner')
        outcomes['conquest' if winner == 0 else ('defeat' if winner == 1 else 'timeout')] += 1
        pop_mins.append(pop_min)
        armies.append(army_max)

    env.close()
    return outcomes, float(np.mean(pop_mins)), float(np.mean(armies))


def make_model_policy(path):
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(path, device='cpu')

    def policy_fn(obs, mask, env):
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        return action if np.ndim(action) else int(action)

    return policy_fn


def make_expert_policy(expert_type):
    state = {}

    def policy_fn(obs, mask, env):
        # one bot per episode: rebuild it whenever the sim clock goes backwards
        if state.get('t', 1e9) > env.sim.game_time:
            state['bot'] = get_opponent(expert_type, faction_id=0)
        state['t'] = env.sim.game_time

        action = state['bot'].act(env.sim.state, env.sim.game_time)
        idx = action_to_index(action.action_type)
        if not mask[idx]:
            idx = int(np.where(mask[:env.action_space_size])[0][-1])   # DO_NOTHING always valid
        if not env.spatial:
            return idx
        # the scripted expert has no spatial brain: it lets the world pick the site
        from agents.district.pretrain_bc import expert_tile
        from simulator.actions import index_to_action
        at = index_to_action(idx)
        tile = expert_tile(env, at) if at.name.startswith('BUILD_') else None
        return np.array([idx, tile if tile is not None else 0], dtype=np.int64)

    return policy_fn


def main():
    p = argparse.ArgumentParser(description='Score a policy (or the scripted expert) by CONQUESTS')
    p.add_argument('--model', type=str, default=None)
    p.add_argument('--expert', type=str, default=None,
                   help="score a scripted bot instead of a model, e.g. 'hard'")
    p.add_argument('--episodes', type=int, default=40)
    p.add_argument('--opponents', type=str, nargs='+', default=['easy', 'medium', 'hard'])
    p.add_argument('--spatial', action='store_true',
                   help='The policy chooses tiles itself (D28)')
    args = p.parse_args()

    if args.model:
        policy_fn, who = make_model_policy(args.model), os.path.basename(args.model)
    elif args.expert:
        policy_fn, who = make_expert_policy(args.expert), f"scripted expert '{args.expert}'"
    else:
        p.error('give either --model or --expert')

    print(f"{who}  |  {args.episodes} episodes per opponent  |  deterministic")
    print(f"{'opponent':10s} {'CONQUESTS':>12s} {'defeats':>9s} {'timeouts':>9s} "
          f"{'pop-min':>9s} {'peak army':>10s}")

    for opp in args.opponents:
        o, pop_min, army = evaluate(policy_fn, opp, args.episodes, spatial=args.spatial)
        print(f"{opp:10s} {o['conquest']:6d}/{args.episodes:<5d} {o['defeat']:9d} "
              f"{o['timeout']:9d} {pop_min:9.2f} {army:10.1f}", flush=True)


if __name__ == '__main__':
    main()
