"""
Behavioral-cloning warm-start for the DISTRICT-tier agent.

The military build (barracks -> train soldiers -> standing army) is a long,
hard-to-explore action sequence. Pure RL never discovers it (barracks built in ~1/10
stochastic episodes) and the policy collapses to economy-only + idle. But we DO have a
scripted policy that solves the task: `AggressiveBot` wins ~9/10 vs the ramping
aggressive opponent. So we:

  1. Collect demonstrations from AggressiveBot playing the agent's slot (faction 0)
  2. Supervised-train the actor to imitate them   (agents.common.bc.behavioral_clone)
  3. Fit the critic to the cloned policy's returns (agents.common.bc.warmup_critic)
  4. Save the warm-started model, optionally RL fine-tuned by agents/district/train.py

Design note (modular architecture):
We clone the DURABLE, transferable skill -- economy -> army PRODUCTION. Combat
resolution is an abstract, swappable oracle (CombatManager); if a spatial tactical
module is added later, the production skill still transfers.

Run (decision_interval and gamma default to the current TIME SCALE — D13):
    python -m agents.district.pretrain_bc --opponent hard \
        --out ./checkpoints/district/district_agent_bc_v14.zip
"""

import os
import sys
import argparse
import numpy as np
import torch

# Allow both `python -m agents.district.pretrain_bc` and direct execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from agents.district.env import RealTimeRTSEnv
from agents.common.extractors import MLPExtractor, MapExtractor
from agents.common.bc import behavioral_clone, warmup_critic, stack_obs
from simulator.opponents import get_opponent
from simulator.actions import action_to_index, index_to_action


def expert_tile(env, action_type):
    """WHERE the expert would put this building (D28).

    The scripted expert has no spatial brain — it names a type and the world finds it the
    best site (D27). That auto-placement is exactly the teacher we want: it is a strong,
    hand-checkable placement policy, so the clone starts by learning to put a lumberyard in
    the woods rather than by discovering it through a million random tiles. What the agent
    can then do that the expert cannot is DEVIATE — and whether deviating is worth anything
    is a question for RL, and for measurement.
    """
    from simulator.managers.building_manager import BuildingManager
    from agents.district.env import PLOT_SIDE

    district = env.sim.state.factions[0].capital
    building_type = action_type.name.replace('BUILD_', '').lower()
    tile, _quality = BuildingManager.find_site(district, building_type)
    if tile is None:
        return None

    ox, oy = env.plot_origin()
    return (tile[0] - ox) * PLOT_SIDE + (tile[1] - oy)


def collect_demonstrations(expert_type='hard', opponent_type='hard',
                           n_episodes=60, seed=0, decision_interval=None,
                           spatial=False):
    """
    Run an expert scripted bot in the agent's slot (faction 0) and record
    (observation, action_index, action_mask) at every decision step.

    The env internally drives faction 1 with `opponent_type`; we override the
    faction-0 action with the expert bot's choice.
    """
    obs_list, act_list, mask_list, tile_relevant = [], [], [], []

    env = RealTimeRTSEnv(opponent_type=opponent_type, decision_interval=decision_interval,
                         spatial=spatial)
    n_actions = env.action_space_size

    wins = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        # Expert controls the agent's faction (0)
        expert = get_opponent(expert_type, faction_id=0)
        done = False
        while not done:
            mask = env.action_masks()
            expert_action = expert.act(env.sim.state, env.sim.game_time)
            action_idx = action_to_index(expert_action.action_type)

            # Safety: if expert picks a masked-invalid action, fall back to a valid one
            if not mask[action_idx]:
                valid = np.where(mask[:n_actions])[0]
                action_idx = int(valid[-1])  # DO_NOTHING is last / always valid

            action_type = index_to_action(action_idx)
            is_build = action_type.name.startswith('BUILD_')

            if spatial:
                tile_index = expert_tile(env, action_type) if is_build else None
                if tile_index is None:
                    # not a build (or nowhere to put it): the tile head is free, and this
                    # sample teaches it nothing
                    is_build = False
                    tile_index = 0
                recorded = [action_idx, tile_index]
                step_action = np.array(recorded, dtype=np.int64)
            else:
                recorded = action_idx
                step_action = action_idx

            obs_list.append({k: v.copy() for k, v in obs.items()} if spatial else obs.copy())
            act_list.append(recorded)
            mask_list.append(mask.copy())
            tile_relevant.append(bool(is_build))

            obs, r, terminated, truncated, info = env.step(step_action)
            done = terminated or truncated   # a timeout truncates; it does not terminate

        if info.get('winner') == 0:
            wins += 1

    env.close()

    placements = int(np.sum(tile_relevant))
    print(f"Collected {len(act_list)} demo steps from {n_episodes} episodes "
          f"(expert CONQUEST rate: {wins/n_episodes:.0%} — a timeout is not a win)")
    if spatial:
        print(f"    of which {placements} are actual PLACEMENTS — the only steps that can "
              f"teach the tile head anything ({placements/len(act_list):.1%})")

    return (
        stack_obs(obs_list),
        np.array(act_list, dtype=np.int64),
        np.array(mask_list, dtype=bool),
        np.array(tile_relevant, dtype=bool),
    )


def make_model(opponent_type, device, decision_interval=None, gamma=0.9995, spatial=False):
    """Create a MaskablePPO model with the SAME architecture / gamma used in RL
    fine-tuning, so the warm-started weights are load-compatible with train.py."""
    def _init():
        return RealTimeRTSEnv(opponent_type=opponent_type, decision_interval=decision_interval,
                              spatial=spatial)
    env = DummyVecEnv([_init])

    if spatial:
        # The plot is a grid, so it gets a convnet; the district summary stays an MLP.
        # MultiInputPolicy is what SB3 calls a policy over a Dict observation.
        policy_kwargs = dict(
            features_extractor_class=MapExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=torch.nn.ReLU,
            share_features_extractor=False,
        )
        policy = "MultiInputPolicy"
    else:
        policy_kwargs = dict(
            features_extractor_class=MLPExtractor,
            features_extractor_kwargs=dict(features_dim=256, hidden_dims=(256, 256)),
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=torch.nn.ReLU,
            # Separate actor/critic extractors: lets the critic warm-up fit returns with its
            # own features WITHOUT perturbing the BC-trained actor.
            share_features_extractor=False,
        )
        policy = "MlpPolicy"

    model = MaskablePPO(
        policy, env, policy_kwargs=policy_kwargs,
        device=device, ent_coef=0.05, gamma=gamma, verbose=0,
    )
    return model, env


def main():
    parser = argparse.ArgumentParser(description='District-tier behavioral cloning warm-start')
    parser.add_argument('--expert', type=str, default='hard')
    parser.add_argument('--opponent', type=str, default='hard')
    parser.add_argument('--episodes', type=int, default=60)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--out', type=str,
                        default='./checkpoints/district/district_agent_bc_v14.zip')
    # decision_interval defaults to None so the env takes it from env_config.json
    # (the single source of truth, derived from the TIME SCALE). gamma tracks the
    # episode length: at 1440 decisions, 0.999 discounts the victory reward to 0.24;
    # 0.9995 keeps it near 0.5. See design_decisions.md D13.
    parser.add_argument('--decision-interval', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=0.9995)
    parser.add_argument('--no-critic-warmup', action='store_true',
                        help='Skip fitting the value function after BC (not recommended)')
    # The decisive actions are the rare ones (ATTACK is <1% of demos). Ablated on
    # conquests, not on accuracy — 0.0: 14/14/10, 0.5: 17/22/16, 1.0: 25/24/18 vs
    # easy/medium/hard. Full inverse frequency wins: it costs aggregate action-match
    # (99% -> 86%) and buys the only decisions that end games (ATTACK recall 27% -> 92%).
    parser.add_argument('--class-weight-power', type=float, default=1.0,
                        help='Re-weight BC samples by (1/class_freq)**p (0 = off)')
    # D28: the agent sees its plot and says WHERE to build, instead of letting the world
    # choose the site for it.
    parser.add_argument('--spatial', action='store_true',
                        help='Train the SPATIAL agent: local-patch obs + BUILD(type, tile)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 70)
    print("DISTRICT AGENT — BEHAVIORAL CLONING WARM-START")
    print("=" * 70)
    print(f"Expert: {args.expert}  |  Opponent: {args.opponent}  |  Device: {device}")
    print(f"Decision interval: {args.decision_interval}s  |  gamma: {args.gamma}")

    # 1. Collect demonstrations
    obs, actions, masks, tile_relevant = collect_demonstrations(
        expert_type=args.expert, opponent_type=args.opponent, n_episodes=args.episodes,
        decision_interval=args.decision_interval, spatial=args.spatial,
    )

    # Report action distribution in demos (sanity: should include barracks + train)
    from simulator.actions import ActionType
    names = [a.name for a in ActionType]
    counts = np.bincount(actions[:, 0] if args.spatial else actions, minlength=len(names))
    top = np.argsort(counts)[::-1][:8]
    print("Demo action distribution (top 8):")
    for i in top:
        if counts[i] > 0:
            print(f"    {names[i]:20s} {counts[i]/len(actions):.1%}")

    # ATTACK is the decisive action and it is RARE — a march is one order, not one per
    # step — so it is called out on its own. Left unweighted, the clone learns to skip it
    # and still scores 99% (see agents/common/bc.py).
    attack_idx = action_to_index(ActionType.ATTACK)
    print(f"    -> ATTACK is {counts[attack_idx]/len(actions):.2%} of demo actions "
          f"({counts[attack_idx]} orders)")

    # 2. Create model + behavioral clone (trains the ACTOR)
    model, env = make_model(args.opponent, device,
                            decision_interval=args.decision_interval, gamma=args.gamma,
                            spatial=args.spatial)
    model = behavioral_clone(model, obs, actions, masks, epochs=args.epochs,
                             class_weight_power=args.class_weight_power,
                             watch_action=attack_idx,
                             tile_relevant=tile_relevant if args.spatial else None)

    # 2b. Critic warm-up (trains the VALUE function to match the BC policy's returns).
    # Without this, RL fine-tuning uses a random critic -> garbage advantages -> wrecks
    # the actor. See docs/design_decisions.md D7.
    if not args.no_critic_warmup:
        def env_factory():
            return RealTimeRTSEnv(opponent_type=args.opponent,
                                  decision_interval=args.decision_interval,
                                  spatial=args.spatial)
        model = warmup_critic(model, env_factory, gamma=args.gamma)

    # 3. Save warm-started model
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.save(args.out)
    env.close()
    print(f"\nSaved BC-pretrained model: {args.out}")
    print("Next (optional) RL fine-tune — only useful if the expert is NOT already optimal:")
    print(f"  python -m agents.district.train --resume {args.out} "
          f"--opponent {args.opponent} --timesteps 500000 --log-dir ./logs_bc_finetune")


if __name__ == '__main__':
    main()
