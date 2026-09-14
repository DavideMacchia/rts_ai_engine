# ML Training — Documentation

Documentation for the RTS AI macro-brain training system (`ml_training/`).

## Read in this order

1. **[architecture.md](architecture.md)** — the big picture: the tiered hierarchy
   (macro brain → civilian districts + military camps → scripted units), which
   tiers learn, unit command ownership, inter-tier communication, policies,
   runtime performance, what exists today, and the roadmap.
2. **[rl_training_system.md](rl_training_system.md)** — how the current training
   pipeline works and how to run it (env, masking, rewards, opponents, behavioral
   cloning, training scripts, config files, commands).
3. **[design_decisions.md](design_decisions.md)** — *why* the system is built this
   way: the diagnostic log of the military-learning problem and the reward
   pitfalls. Read this before changing rewards, the opponent, or the horizon.
4. **[roadmap.md](roadmap.md)** — **the current plan**: the steps and objectives
   toward the full multi-model hierarchy (macro orchestrator + civil-RL and
   military-RL district specialists), and the locked design decisions. This
   supersedes the tier plan in architecture.md and the older
   `roadmap_war_economy_macro.md`.

## One-line status

**Infrastructure works; the agent is mid-redesign and weak.** An earlier tier reached 10/10
deterministic wins against every scripted bot (the `bc_v1` era, now retired — those checkpoints
predate the current action space and no longer load). The **district-unified** redesign — one
tier doing economy *and* military, scored by a graded valuation of its territory — grew the
action space (30 → 37 actions) and reintroduced the project's oldest failure: the agent
**doesn't build military**. A BC + critic-warm-up policy on the current env (train one with
`pretrain_bc` — checkpoints are not committed, they regenerate in minutes) loads and runs but
is passive: it clones a scripted expert that itself idles ~86% and conquers ~38%, so it builds
almost no army and loses most games. Restoring military play under the new objective is the
active work (see design_decisions).
