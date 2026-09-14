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

## Cross-cutting constants (always true)

- **C1 — Reliable measurement.** Every change is judged on the conquest metric, with the
  opponent held fixed and seeds controlled. Never ablate on a proxy.
- **C2 — Retrain on every feature.** Any new action or observation invalidates prior policy
  weights, so the affected models are retrained. The infrastructure (env, action masking,
  BC + critic warm-up, the pipeline) transfers; the weights do not.

---

## Steps and objectives

Each step is a milestone with a definition of "done". No timing, no methods.

**Step 0 — Reliable measurement.**
A setup that can say with confidence whether a change helped or hurt. Precondition for all below.

**Step 1 — Territory: bounded square claims.**
A district, when founded, immediately claims a **square** area of the map with a **calibratable
side (min/max)**. Territories **cannot overlap**. The claim is delimited from founding, even with
nothing built. Economy buildings pack inside it; the finite space is the growth limit.
*Map role: constraint, with an explicit owned boundary.*

**Step 2 — Civil specialist (RL).**
A district that, on its own, reliably develops a full economy inside its territory — climbs the
production chain to high grade, holds a stable population, and staffs what it builds —
**measurably better than any scripted teacher**.

**Step 3 — Military specialist + military organization (RL).**
A separate model that raises and **employs** an army effectively, together with the definition of
**what a military district/camp is and how its forces are organized, commanded, and positioned**.
This is the precondition for sieges.

**Step 4 — Spatial / structural defense (sieges).**
Combat stops being a faction-scale oracle and becomes **spatial**: an attack marches to a point,
and **walls / gates / towers gain calibrated defensive function** (walls block and channel, gates
are the breach points, towers cover a radius). From here **both** district types **place defensive
structures on their boundary as a spatial decision**, and it matters. *Deferred to here by design:
it needs the military organization from Step 3.*
*Map role: decision (the defensive layer) for both district tiers; economy stays packing.*

**Step 5 — Guns-vs-butter in one district.**
The civil and military specialists coexist governing a single territory that **develops AND
defends/fights** without collapsing onto one half — the milestone proving specialization beats the
unified brain, now including fortification.

**Step 6 — Multi-district worlds.**
The simulator instantiates and runs **≥2 non-overlapping districts per faction** (found, own, lose
them), so coordination and competition for territory become real.

**Step 7 — Macro orchestrator (RL).**
A model that decides **where to found districts** (placing non-overlapping square claims across
terrain, deposits, and fronts), how to **allocate people and forces** between them, and empire
priorities — **measurably better than districts acting independently**.

**Step 8 — Full stack + self-play.**
The complete hierarchy (macro → civil + military districts → units) wins convincingly against
strong scripted opponents, then against **itself (self-play)** to remove the scripted-opponent
ceiling. The final result.

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
