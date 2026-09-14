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

## One-line status

**Stage 0 complete.** The **district-tier** agent plays economy *and* military: the deliverable
`checkpoints/district/district_agent_bc_v1.zip` (behavioral cloning + critic warm-up) wins 10/10
deterministic games against every scripted opponent. RL fine-tuning is currently
*not* useful — the scripted expert is already optimal, so there is no headroom
(design_decisions §D8). Next up: the **`Faction` → `District` refactor** (architecture.md §7 Stage 1),
which unlocks the macro tier. Self-play is deferred (§D9).
