# Roadmap — to the full multi-model hierarchy

The end state is an empire-level AI as a **set of specialized models**, not one flat brain:
a **macro orchestrator** that coordinates multiple **districts**, each district itself governed
by two specialists — a **civil** economy brain and a **military** brain — over scripted units.

**Why specialization, not one brain.** The project has twice tried to make a single agent do
both economy and war, and twice it collapsed onto one objective: pre-D32 it built only military
and never developed; the later *district-unified* redesign did the mirror — it develops the
economy and never fights (measured: army ~0.6, ~3/30 conquests). A single scalar reward over
"build a civilisation" *and* "win a war" collapses onto whichever is easier to optimise.
Specialization sidesteps that. See [design_decisions.md](design_decisions.md) (esp. D32).

**Civil is a first-class learning problem, not a script.** The civil economy is at least as
complex as the military one (the whole D29–D37 arc is evidence), and a scripted teacher is a
ceiling behavioral cloning cannot exceed. So the civil district is RL, not a utility bot. This
supersedes the earlier "only the adversarial tiers learn" plan in
[architecture.md](architecture.md).

---

## Current state relative to this roadmap (2026-09-14)

The **infrastructure** is built and works end to end: the simulator, the environment (action
masking, graded reward, an `economy`/`conquest` objective switch, an optional spatial mode), the
training pipeline (behavioral cloning + critic warm-up → MaskablePPO), and evaluation.

Against the roadmap steps:

- **Step 0 (measurement)** — *partially there.* Seeded reproducibility and a conquest-based
  `evaluate` exist; the confidence-interval protocol below is not yet standard practice.
- **Step 1 (territory)** — *not started.* Today a district is a **square plot of fixed radius**
  (`DISTRICT_RADIUS`), not an adjustable, explicitly-claimed, non-overlapping territory; a world
  still starts with one district.
- **Step 2 (civil RL)** — *not started.* The only trained model is the **unified** district agent
  (`objective='conquest'`), which collapses onto economy and is weak (v18: ~3/30, army ~0.6). The
  `objective='economy'` path and the scripted `SettlerBot` exist as precursors, not a validated
  specialist that beats its teacher.
- **Steps 3–8** — *not started.* `agents/camp/` and `agents/macro/` are empty placeholders;
  combat is a faction-scale oracle; walls/gates/towers are buildable but inert.

**In one line:** we are at the pre-Step-1 stage — solid infrastructure plus a unified district
agent that is exactly the thing this roadmap moves *away from*.

---

## Cross-cutting constants (always true)

- **C1 — Reliable measurement.** Every change is judged on the conquest metric, with the
  opponent held fixed and seeds controlled. Never ablate on a proxy.
- **C2 — Retrain on every feature.** Any new action or observation invalidates prior policy
  weights, so the affected models are retrained. The infrastructure (env, action masking,
  BC + critic warm-up, the pipeline) transfers; the weights do not. A feature is not "done"
  until the retrained model matches or beats the prior baseline on the shared metric.

### Evaluation protocol (makes every threshold below meaningful)

Every headline number: **≥100 seeded games per opponent**, opponent **held fixed** at the stated
difficulty, reported as **mean ± confidence interval**. A result counts as an *improvement* only
when its advantage over the baseline has **non-overlapping CIs** (≈ more than 2 standard errors).
Primary metric = **conquests** where there is an enemy; where there is none (the civil district),
the metric is the **graded economy score** (people by skill tier + goods by chain depth).

> The percentage thresholds in the steps below are **proposals, to be calibrated against the
> noise band that Step 0 establishes.** What is non-negotiable is the *shape*: a metric, a
> threshold above noise, and — wherever a step claims "it works" — an **ablation** that beats a
> baseline.

---

## Steps and objectives

Each step is a milestone with a definition of "done". No timing, no methods.

**Step 0 — Reliable measurement.**
A setup that can say with confidence whether a change helped or hurt. Precondition for all below.
**Done when:** same seed → **bit-identical** trajectory (hash), and the harness resolves a true
Δ of ~10 conquest points as significant at the chosen *n* (it can tell a known-better policy from
a known-worse one).

**Step 1 — Territory: bounded square claims.**
A district, when founded, immediately claims a **square** area with a **calibratable side
(min/max)**, territories **cannot overlap**, and the claim is delimited from founding.
*Map role: constraint, with an explicit owned boundary.*
**Done when:** over **≥1000 random foundings, 0 violations** (side in [min,max], within bounds,
**0 overlapping tiles** between claims); an ablation shows a **monotonic** relation (smaller side →
lower max economy score, i.e. space *is* the binding limit); and at side = current radius the
civil metric is **statistically unchanged** vs the pre-territory baseline.

**Step 2 — Civil specialist (RL).**
A district that, on its own, reliably develops a full economy inside its territory —
**measurably better than any scripted teacher**.
**Done when:** in **≥90% of seeds** it staffs the full iron-tier trades (L2/L3); **0% extinction**
over ≥100 seeds; population holds near the plot's carrying capacity; idle share **≤ ~20%** at
maturity; and its graded economy score is **strictly above the best scripted teacher** by a margin
beyond the noise band (i.e. it is not capped by the script).

**Step 3 — Military specialist + military organization (RL).**
A separate model that raises and **employs** an army effectively, plus the defined **military
organization** (what a camp is; how forces are composed, commanded, positioned).
**Done when:** from a fixed economic starting state it actually uses its army (ATTACK used, peak
army ≥ threshold, not ~0) and wins — **conquest ≥ ~70% vs easy / ≥ ~50% vs hard** over ≥100 seeds —
with the outcome attributable to the defined organization.

**Step 4 — Spatial / structural defense (sieges).**
Combat becomes **spatial**; walls / gates / towers gain calibrated defensive function; both
district types **place** defensive structures on their boundary as a spatial decision.
*Map role: decision (the defensive layer) for both tiers; economy stays packing.*
**Done when (fortification pays — the mirror of the D28 null result):** removing or worst-placing
walls/gates/towers **drops the defender's survival by ≥15 points** over ≥100 seeds; behind walls
the defender's casualty ratio improves by a stated factor and gates are the breach points; and a
policy that **chooses** placement **beats** a fixed/random baseline on survival, margin > noise.

**Step 5 — Guns-vs-butter in one district.**
The civil and military specialists coexist governing one territory that **develops AND
defends/fights** without collapsing onto one half.
**Done when:** in the *same* game the economy score is **within ~15%** of the Step-2 civil
specialist **and** the conquest rate is **within ~15%** of the Step-3 military specialist; no
collapse (peak army above threshold **and** high-grade trades staffed) in **≥90% of seeds**.

**Step 6 — Multi-district worlds.**
The simulator instantiates and runs **≥2 non-overlapping districts per faction** (found, own, lose).
**Done when:** games complete with ≥2 non-overlapping districts per faction and **0 invariant
violations** over ≥K seeds; and the **"independent districts" baseline** (no coordination) has a
measured conquest rate vs fixed opponents — the number Step 7 must beat.

**Step 7 — Macro orchestrator (RL).**
A model deciding **where to found** districts, how to **allocate** people/forces, and empire
priorities.
**Done when:** it **beats the independent-districts baseline** (Step 6) in conquests, margin >
noise, ≥100 seeds, opponent fixed; and an ablation shows freezing it (random founding/allocation)
**drops** conquests significantly (where-to-found and allocation are worth something).

**Step 8 — Full stack + self-play.**
The complete hierarchy wins against strong scripted opponents, then against **itself**.
**Done when:** **conquest ≥ ~80% vs the strongest scripted opponent** over ≥100 seeds; and the
self-play-trained stack **beats the BC/scripted-ceiling stack head-to-head ≥ ~65%** over ≥100
games (opponent = the previous version, held fixed).

---

## Dependencies

- C1 and C2 hold throughout.
- 0 and 1 are foundational.
- 2 and 3 follow 1 (they need a territory).
- **4 follows 3** (sieges need the military organization).
- 5 follows 2 + 4.
- 6 follows a working district (2 / 3).
- 7 follows 5 + 6.
- 8 follows 7.

---

## Locked design decisions (this iteration)

- **Square map grid; square territory with an adjustable side (min/max).** Not hexagonal.
- **Districts cannot overlap.**
- **Territory is claimed at founding** and delimited even when empty.
- **Civil district is RL**, full complexity — not a scripted utility bot.
- **Two district specialists** (civil + military), same spatial mechanic, different building sets.
- **Within a district there are two kinds of placement:** economic (packing, world-managed) and
  **defensive (walls/gates/towers on the boundary — a spatial decision the model learns)**.
- **Sieges (spatial combat) come after** the military tier and its organization are defined.
- **Retrain on every feature** (C2).
