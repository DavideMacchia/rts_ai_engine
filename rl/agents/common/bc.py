"""
Behavioral cloning + critic warm-up — tier-agnostic.

Both stages are reusable by ANY agent tier (district, macro, camp). They only assume:
- a MaskablePPO model whose policy exposes `evaluate_actions(obs, actions, action_masks)`
- an environment exposing `action_masks()` and the Gymnasium step/reset API

Why both stages are required
----------------------------
`behavioral_clone()` trains the **actor** only (its loss is `-log pi(a_expert | s)`).
The value head never receives a target, so it stays at random init. Fine-tuning PPO
with that random critic yields advantages `A = return - V(s)` that are huge and
uniformly positive, which wrecks the cloned policy. `warmup_critic()` fixes the critic
BEFORE any RL fine-tune. See docs/design_decisions.md D7.
"""

import numpy as np
import torch


# --- observations may be a flat vector OR a dict (the spatial agent, D28) ---

def stack_obs(obs_list):
    """Turn a list of observations into arrays, whichever shape they are."""
    if isinstance(obs_list[0], dict):
        return {k: np.array([o[k] for o in obs_list], dtype=np.float32) for k in obs_list[0]}
    return np.array(obs_list, dtype=np.float32)


def _to_torch(obs, device):
    if isinstance(obs, dict):
        return {k: torch.as_tensor(v, device=device) for k, v in obs.items()}
    return torch.as_tensor(obs, device=device)


def _take(obs_t, idx):
    if isinstance(obs_t, dict):
        return {k: v[idx] for k, v in obs_t.items()}
    return obs_t[idx]


def _len(obs_t):
    return len(next(iter(obs_t.values()))) if isinstance(obs_t, dict) else len(obs_t)


def behavioral_clone(model, obs, actions, masks, epochs=15, batch_size=512, lr=3e-4,
                     class_weight_power=0.5, watch_action=None, tile_relevant=None):
    """Supervised training of the ACTOR to imitate demonstrations.

    Loss = masked cross-entropy = -log pi(expert_action | obs), weighted by class.

    Why weighted
    ------------
    The demonstrations are wildly imbalanced, and the rare actions are the DECISIVE ones.
    The expert orders ATTACK a handful of times per episode (a march is one order, not one
    per step) — under 1% of the samples — while it builds and trains in all the others. An
    unweighted clone reaches 99% action-match by never attacking at all: it copies the economy
    perfectly, hoards an army, and conquers nothing (D25). Accuracy is not the objective; the
    objective is the 1% of decisions that win the game.

    `class_weight_power` p re-weights each sample by (1/freq(a))**p, normalised to mean 1:
        p = 0    unweighted — this is what produces the pacifist clone
        p = 0.5  sqrt-inverse frequency — amplifies the rare action without letting a
                 handful of samples dominate the gradient
        p = 1    full inverse frequency — every action class contributes equally

    Args:
        model: MaskablePPO model
        obs, actions, masks: demonstration arrays (N, obs_dim), (N,), (N, n_actions)
        watch_action: index of a pivotal action to report RECALL for. Aggregate accuracy
            hides exactly the failure this function exists to prevent, so we print the
            share of the expert's ATTACK orders the clone actually reproduces.
    """
    device = model.device
    obs_t = _to_torch(obs, device)
    act_t = torch.as_tensor(actions, device=device)
    mask_t = torch.as_tensor(masks, device=device)

    #: Two heads (what to build, where) rather than one (D28).
    factored = act_t.ndim == 2
    types = actions[:, 0] if factored else actions

    n = _len(obs_t)
    counts = np.bincount(types, minlength=int(types.max()) + 1).astype(np.float64)
    freq = counts[types] / n
    weights = (1.0 / np.maximum(freq, 1e-9)) ** class_weight_power
    weights = weights / weights.mean()
    w_t = torch.as_tensor(weights, dtype=torch.float32, device=device)

    # The tile head is only meaningful where the expert actually BUILT something. On every
    # other step its choice was ignored by the environment, so training it on that step
    # would be fitting noise — and there are ~50 such steps for every real placement.
    if factored:
        if tile_relevant is None:
            tile_relevant = np.ones(n, dtype=bool)
        tile_t = torch.as_tensor(np.asarray(tile_relevant, dtype=np.float32), device=device)

    optimizer = torch.optim.Adam(model.policy.parameters(), lr=lr)

    print(f"\nBehavioral cloning: {n} samples, {epochs} epochs, batch {batch_size}, "
          f"class_weight_power={class_weight_power}"
          + (f", FACTORED heads (tile trained on {int(tile_relevant.sum())} placements)"
             if factored else ""))
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        correct = 0
        watch_hit = watch_total = 0
        tile_hit = tile_total = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            b_obs, b_act, b_mask, b_w = _take(obs_t, idx), act_t[idx], mask_t[idx], w_t[idx]

            dist = model.policy.get_distribution(b_obs, action_masks=b_mask)

            if factored:
                heads = dist.distributions             # one Categorical per head
                lp_type = heads[0].log_prob(b_act[:, 0])
                lp_tile = heads[1].log_prob(b_act[:, 1])
                # the tile term is switched off wherever no building was placed
                log_prob = lp_type + tile_t[idx] * lp_tile
            else:
                log_prob = dist.log_prob(b_act)

            loss = -(log_prob * b_w).mean() - 0.01 * dist.entropy().mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(idx)

            # Accuracy: does the policy's argmax match the expert?
            with torch.no_grad():
                dist = model.policy.get_distribution(b_obs, action_masks=b_mask)
                if factored:
                    heads = dist.distributions
                    pred = heads[0].probs.argmax(dim=1)
                    pred_tile = heads[1].probs.argmax(dim=1)
                    is_build = tile_t[idx].bool()
                    tile_total += int(is_build.sum().item())
                    tile_hit += int((is_build & (pred_tile == b_act[:, 1])).sum().item())
                    truth = b_act[:, 0]
                else:
                    pred = dist.distribution.probs.argmax(dim=1)
                    truth = b_act

                correct += (pred == truth).sum().item()
                if watch_action is not None:
                    is_watch = truth == watch_action
                    watch_total += int(is_watch.sum().item())
                    watch_hit += int((is_watch & (pred == truth)).sum().item())

        line = (f"  epoch {epoch+1:2d}/{epochs}  loss={total_loss/n:.4f}  "
                f"action_match={correct/n:.1%}")
        if watch_total:
            line += f"  ATTACK recall={watch_hit/watch_total:.1%}"
        if tile_total:
            line += f"  tile_match={tile_hit/tile_total:.1%}"
        print(line)

    return model


def warmup_critic(model, env_factory, gamma,
                  n_episodes=40, epochs=30, batch_size=512, lr=1e-3, seed=10000):
    """Fit ONLY the value function to returns under the (already cloned) policy.

    Rolls out the current policy, computes discounted returns-to-go, and regresses the
    value head to them WITH THE ACTOR FROZEN: we optimize only critic-branch params.
    With `share_features_extractor=False` the critic owns its `vf_features_extractor`,
    so the actor's action distribution is provably unchanged.

    Watch `explained_var` — it should reach ~0.9. (With a *shared* frozen extractor it
    plateaus near 0.29: the critic would have to predict values from features optimized
    to pick actions.)

    Args:
        model: MaskablePPO model (post behavioral cloning)
        env_factory: zero-arg callable returning a fresh env exposing `action_masks()`
        gamma: discount used to compute returns (must match the fine-tune gamma)
    """
    device = model.device

    # 1. Collect rollouts of the current policy (stochastic, for state coverage)
    env = env_factory()
    obs_list, ret_list = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_obs, ep_rew = [], []
        while not done:
            mask = env.action_masks()
            a, _ = model.predict(obs, deterministic=False, action_masks=mask)
            ep_obs.append({k: v.copy() for k, v in obs.items()}
                          if isinstance(obs, dict) else obs.copy())
            obs, r, terminated, truncated, _ = env.step(a if np.ndim(a) else int(a))
            done = terminated or truncated   # a timeout truncates; it does not terminate
            ep_rew.append(r)
        G = 0.0
        rets = [0.0] * len(ep_rew)
        for i in reversed(range(len(ep_rew))):
            G = ep_rew[i] + gamma * G
            rets[i] = G
        obs_list.extend(ep_obs)
        ret_list.extend(rets)
    env.close()

    obs_t = _to_torch(stack_obs(obs_list), device)
    ret_t = torch.as_tensor(np.array(ret_list, dtype=np.float32), device=device)
    n = _len(obs_t)

    # 2. Optimize ONLY the critic branch -> the actor is untouched.
    value_params = list(model.policy.value_net.parameters()) + \
                   list(model.policy.mlp_extractor.value_net.parameters())
    if not model.policy.share_features_extractor:
        value_params += list(model.policy.vf_features_extractor.parameters())
    optimizer = torch.optim.Adam(value_params, lr=lr)

    print(f"\nCritic warm-up: {n} samples (returns mean {ret_t.mean():.0f}), "
          f"{epochs} epochs, {len(value_params)} value tensors")
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            values = model.policy.predict_values(_take(obs_t, idx)).squeeze(-1)
            loss = torch.nn.functional.mse_loss(values, ret_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        if epoch % 5 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                pred = model.policy.predict_values(obs_t).squeeze(-1)
                ev = 1 - (ret_t - pred).var() / (ret_t.var() + 1e-8)
            print(f"  epoch {epoch+1:2d}/{epochs}  mse={total/n:.1f}  "
                  f"explained_var={ev.item():.3f}")

    return model
