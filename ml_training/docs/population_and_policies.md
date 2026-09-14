# Population, Policies, and the District's View

> **Current plan: [roadmap.md](roadmap.md).** The "10/10 BC oracle" referenced below is
> historical — it was retired by the district-unified redesign (new action space and objective).
> The population/policy mechanics documented here still stand.

> Status: design document. **§2–§5 are implemented in the training simulator**
> (`simulator/policies.py`, `District.calculate_happiness()`,
> `PopulationManager.fertility()`, `TrainingManager.conscript_district()`), covered by
> `tests/unit/test_policies.py`. §6 (spatial view) and §7 (cohort model) are not.
> The Rust game is untouched — it is still under construction, so the model leads.
>
> **All defaults reproduce the un-governed district exactly.** Verified by a
> bit-identical trajectory hash against the pre-district code over 5 episodes ×
> 700 steps × 2 factions, and by the BC oracle: still 10/10 vs all three bots.
> Turning a policy *on* changes behaviour; that is the point. Turning none on
> changes nothing, which is why the regression oracle survives.
> Companion to [architecture.md](architecture.md) (the tiered agent) and
> [design_decisions.md](design_decisions.md) (why past choices were made).
> Game-side references: `documentation/gameplay/population_system.md`,
> `documentation/gameplay/area_system.md`, `documentation/gameplay/map_system.md`.

This document answers three coupled questions:

1. Sims manage themselves. So **how does a district influence them?** (§1–§5)
2. What must the district brain **see and do** — its map, deposits, and where it
   places buildings? (§6)
3. How do we **simulate autonomous sims cheaply enough to train on**? (§7)

---

## 1. The bug at the root: a policy is a price with no effect

`Area` in the Rust game *is* the `District` we built in Stage 1 — its own comment
says "represents a district/settlement". It already carries ten policies
(`src/components/areas/area_policies.rs`). And the only system that reads them,
`src/systems/areas/policies.rs`, is titled:

```rust
/// Policies decrease happiness over time
pub fn apply_policy_effects(...)
```

It computes `calculate_happiness_modifier()` and subtracts happiness. That is all it
does. The training simulator mirrors it exactly: `POLICY_EFFECTS` in
`simulator/config.py` is a dict of happiness maluses that **no code reads at all**.

> **A policy is, by construction, a pair (effect, cost). Today only the cost exists.**

`forced_conscription` does not conscript anyone. `minimum_food` does not ration
anything. `population_rate: Increase` does not cause a single extra birth. They make
sims sad, and nothing else. Every one of them is a `-15.0` looking for a verb.

This document supplies the verbs.

---

## 2. Four levers, not one

"Policy" is currently one word for four mechanisms that differ in **how much
autonomy they take from the sim**. Naming them separately is the whole design.

| # | Lever | Changes… | Sim still chooses? | Cost |
|---|---|---|---|---|
| **L1** | **Affordance** | what is *possible* | yes, fully | none (indirect) |
| **L2** | **Incentive** | what is *attractive* | yes | low |
| **L3** | **Prohibition** | the *option set* (masks an action) | among what's left | medium |
| **L4** | **Coercion** | *overrides* the choice | no | high |

L1 and L2 **preserve** the sim's agency: they reshape the decision landscape and
the decision stays the sim's own. L3 and L4 **suspend** it.

The existing `AreaPolicies` already contains one of each — they were designed
correctly and only lack implementation:

| Existing policy | Lever | What it must actually do |
|---|---|---|
| `minimum_food` | **L1** affordance | ration the food a sim may draw per meal; it still *chooses* to eat |
| `population_rate: Increase/Decrease` | **L2** incentive | shift the fertility term (see §5) |
| `elders_retirement` | **L2/L4** | change the elder cohort's job-seeking weight to 0 |
| `automatic_mode` | — | not a policy: it toggles *who* sets the policies (district AI vs player) |
| `block_families` | **L3** prohibition | mask `Court`/`Marry` out of the sim's action set |
| `cant_leave` | **L3** prohibition | mask any action whose target lies outside the area |
| `siege_mode` | **L1 + L3** | close the area boundary; ration; mask outside-work |
| `force_eat` | **L4** coercion | override the current action with `Eat` |
| `force_sleep` | **L4** coercion | override the current action with `Rest` |
| `forced_conscription` | **L4** coercion | reassign `profession` and transfer unit ownership |

Note the price list already encodes the hierarchy: `block_families` (L3) costs −10,
`forced_conscription` (L4) costs −25. **Cost rises with coerciveness.** Keep it that
way; it is the balancing invariant that makes the four levers a real trade-off rather
than a menu.

---

## 3. Where the levers enter: the sim's utility

The district never sends a sim a command. It **publishes a context**, and the sim
reads that context when it scores its own options.

```
Utility(action) = need_deficit(action)          # how badly do I want this?
                × base_weight(action)           # personality / innate attributes
                × district_modifier(action)     # L2 incentive   (continuous, > 0)
                × available(action)             # L1 affordance  (0 or scarcity-scaled)
                × allowed(action)               # L3 prohibition (0 or 1)

chosen = argmax Utility(action)                 # unless L4 overrides
```

Three of the four levers are **factors in this product**. Coercion (L4) is the one
that is *not*: it does not multiply a utility, it replaces the choice, or mutates the
sim's `profession` and ownership. That asymmetry is the honest signature of coercion,
and it is why it must cost the most.

**`allowed(action)` is the same primitive as `action_masks()`** in
`agents/district/env.py`. One masking mechanism, two tiers: it masks illegal actions
for MaskablePPO, and forbidden actions for the sim. Implement it once.

### Why publish a context instead of issuing orders

- **O(1), not O(N).** The district writes a handful of modifiers regardless of whether
  it holds 20 sims or 2000. It never enumerates them.
- **It degrades, it does not break.** A district under siege, or one whose brain is
  starved of CPU that frame, still has sims that eat and work.
- **No dual command.** A sim always has exactly one owner (architecture §5). Context
  is not command, so a camp and a district cannot fight over the same sim.

---

## 4. Invariants

These are the rules that make the difference between a world and a spreadsheet.

**I1 — Prohibition never masks a vital need. Coercion may force one.**
Masking `Eat` leaves the utility AI with no fallback: sims deadlock and die en masse.
Masking `Socialize`, `Study`, `Court` is fine — they are discretionary, the sim falls
back on something else and pays in happiness.
This is why `minimum_food` is designed correctly *in its name*: it is **rationing**
(L1, a constraint on the resource) and not a ban on eating. The sim still chooses to
eat; it finds less food. The unhappiness follows from hunger, not from a decree.
Symmetrically, `force_eat` (L4) is safe: forcing a vital need cannot deadlock.

> Forbidding a vital need is lethal. Forcing one is merely unpopular.

**I2 — Policies act on conditions. They never overwrite outcomes.**
The tempting implementation of `population_rate: Increase` is
`pregnancy_chance *= 1.5`. Do not. You get a number, not a world: nobody can explain
*why* births rose, ten such multipliers compose into nonsense, and balancing becomes a
table of fudge factors. Make fertility depend on food, housing, health, happiness and
safety — then a pro-natalist policy is a *subsidy* that moves resources to families,
and the demography **responds**. Conditions compose; multipliers do not.

**I3 — Every policy is (effect, cost). A policy with only a cost is a bug.**
This is the current state of all ten. Any new policy must name the term of the utility
product, the resource constraint, or the mask that it touches.

**I4 — Ownership follows scope** (architecture §6). A policy is an action in the
action space of the brain whose scope it affects:

| Policy | Scope | Owner | Learned? |
|---|---|---|---|
| `minimum_food`, `population_rate` | district | civilian district | **no** — utility rules |
| `siege_mode`, `forced_conscription` | district under threat | civilian district | no |
| `block_families`, `cant_leave` | camp | **military camp** | **yes** — RL |
| empire-wide laws | global | **macro** | **yes** — RL |

Only adversarial tiers learn (architecture §1, and D8: where a rule captures the
optimum, the rule beats RL). Civilian district policies are rules. Camp policies are
learned, because their optimum depends on what the enemy does.

**I5 — The district brain observes nothing the training simulator cannot produce.**
See §7. Violating this is how a policy that wins in training loses in the game.

---

## 5. Case study: incentivising births

Today, birth is nobody's decision: `create_random_marriages` is
`rng.random_bool(marriage_chance)` over eligible adults in the same area, and
`create_pregnancies` is `rng.random_bool(0.05)` over married couples. There is no
utility to shift, so **L2 cannot reach it**. This is the concrete reason "are policies
enough?" is currently answered *no*.

Two ways out.

**Rejected (I2):** `population_rate: Increase → pregnancy_chance × 1.5`.

**Adopted:** make fertility a *conditional* propensity of the couple.

```
fertility(couple) = base
                  × f_food(food_security_of_district)      # can we feed a child?
                  × f_housing(free_capacity)               # is there room?
                  × f_health(min(life_points))
                  × f_happiness(mean(happiness))
                  × policy_modifier(population_rate)       # L2, the ONLY direct term
```

`population_rate: Increase` then does two things: it applies a modest L2 modifier, and
— more importantly — it makes the district's *rules* prioritise housing and food
surplus, i.e. it changes the **conditions**. `Decrease` conversely relaxes housing
priority and may add an L3 mask on `Court`.

The payoff: births now rise because the district is prosperous and has room, which is
what the player will believe is happening anyway. And the same machinery gives
`block_families` (L3 mask on `Court`) for free, with no special-casing.

> **Timing warning for RL.** A pro-natalist policy pays off after `kid_duration`
> (300 s in `game_data/person.json`) — the newborn is not a worker for a long time.
> That is precisely the credit-assignment horizon that cost us D2 and D6. The
> delta reward on population growth fires immediately, so the signal exists; verify
> it empirically **before** putting the policy in a learned action space, not after.

---

## 6. The district's view: local map, deposits, placement

The district brain must see where it is building. This introduces a **second spatial
scale**, and the two must not be confused:

| Scale | Owner | Structure | Actions |
|---|---|---|---|
| **Region graph** (architecture §4) | macro | nodes = regions, edges = distance/terrain | `FOUND_DISTRICT(region)` |
| **District plot map** (this doc) | district | local tile grid, `TileType`, deposits | `BUILD(type, tile)` |

The game already has the lower scale: `resources/map.rs` is a tile grid (`width`,
`height`, `tile_data: Vec<Vec<TileType>>`) with `TileType ∈ {Grass, Water, Mountain,
Forest, Sand, Snow, Road, Empty}` and `get_neighbors()`. Resource veins are sketched
in `src/spawners/resources/veins.rs` — currently **entirely commented out**, so
deposits do not exist yet. They are a prerequisite for this section.

### Observation

The district's view becomes a **local patch** around its warehouse, not a flat vector:

- per tile: terrain one-hot, deposit type + richness, occupied-by-building, distance
  to warehouse, in-district flag;
- per district: the current 22 scalars (stock, population, buildings) as a global
  summary token.

### Action

Building placement becomes a **pointer action**, exactly as `FOUND_DISTRICT(region)`
is at the macro tier:

```
BUILD(building_type, tile)
```

Do **not** flatten this into `Discrete(16 × W × H)`: it explodes, and it does not
generalise across map sizes. Factor it — one head over building types, one head over
tiles — and mask both:

- a tile is maskable if it is empty, in-district, and terrain-compatible;
- a building type is maskable if the district can afford it (already implemented).

This is where the retained `TransformerExtractor` finally becomes the right tool: a
variable-size set of tiles with spatial structure, rather than 22 flat scalars where
we correctly replaced it with an MLP.

> **Consequence.** This changes both the observation and the action space, so
> `district_agent_bc_v1.zip` does **not** transfer (it is tied to `(22,)` and
> `Discrete(20)`). Retraining is ~3 min. Note also that SB3 stores the extractor's
> module path inside the `.zip`: do not move `agents/common/extractors.py`.
> Keep the current flat agent as the regression baseline until the spatial one beats it.

---

## 7. Simulating autonomous sims — cheaply enough to train on

This is the hard constraint, and it deserves to be stated bluntly:

> **We will not train RL on an agent-based demography.**
> ~720 decisions per episode × thousands of sims × millions of steps is intractable,
> and the per-sim noise buries the victory signal we already fought to hear (D6).

The game's ECS **is** the ground truth. The training simulator is a fast
approximation of it (architecture §8). The job is not to make the training sim
agent-based; it is to make its **aggregate behaviour match** the agent-based one.

### The fidelity ladder

| Level | Model | Cost | Used for |
|---|---|---|---|
| **L0** | `population: int` + Bernoulli birth | ~free | **today** |
| **L1** | **cohorts × profession** | ~free | **proposed target** |
| **L2** | sampled individuals | expensive | calibration & validation only |
| **L3** | full ECS (the game) | very expensive | ground truth |

**L0 lies, and we can name the lie.** `get_available_workers()` is
`population − military`: every head is a working adult. But `person.json` sets
`kid_duration: 300` and `adult_duration: 600`, so a large fraction of a real
population cannot work. The BC policy was trained believing it has ~50% more
labour than it will have. The 10/10 is honest *about the simulator*; it is the first
thing that will break on transfer to the game.

**L1 is the honest minimum.** Replace the scalar with compartments:

```python
@dataclass
class Population:
    kids:   int                     # cannot work
    adults: dict[Profession, int]   # the labour force, by profession
    elders: int                     # reduced productivity (elder_productivity in person.json)
    happiness: float                # aggregate — the variable POLICY_EFFECTS is waiting for
```

Flows between compartments (ageing, births, deaths, profession changes) are
rate equations, not individuals. This is cheap, it stops lying about the labour force,
and — critically — it gives `happiness` a home, which is what makes L1/L2/L3 policies
expressible at all.

### Jobs fill themselves

`District.calculate_worker_assignments()` currently sorts buildings by
`BUILDING_PRIORITIES` and pours workers in. That is a central planner, and it is
exactly what §3 says the district must *not* be.

At L1, model job filling as **attractiveness-weighted matching**, the aggregate shadow
of sims choosing work by utility:

```
share(building) ∝ attractiveness(building) × slots_free(building)
```

where `attractiveness` folds in wage/food access, distance from housing, profession
match (`BuildingWorkers.required_profession` already exists), and the district's
L2 incentives. Understaffing then *emerges* from unattractive work, instead of being
a `min(needed, remaining)` truncation. The district influences staffing by making work
attractive — the same lever it uses on everything else.

### Calibration protocol (the step nobody does)

L1's rate constants are not to be invented. `BIRTH_CHANCE_PER_TURN = 0.1` and
`MIN_FOOD_PER_CAPITA_FOR_GROWTH = 2.0` are made-up numbers today.

1. Instrument the Rust ECS to log, per area per tick: cohort sizes, food per capita,
   happiness, births, deaths, staffing ratios.
2. Sweep the game across a grid of conditions (food surplus/deficit, housing
   pressure, each policy on/off).
3. **Fit** the L1 rate equations to those curves.
4. Gate on error: L1 must track L3's cohort trajectories within a stated tolerance
   over a full episode length. Add it to CI as a fidelity test.

**Sim-to-real fidelity lives in these constants, not in the number of observation
features.** A 200-feature observation over a mis-calibrated model transfers worse
than a 22-feature one over a calibrated model.

### Invariant I5, restated

The district brain may only observe quantities that L1 produces. If a feature exists
in the ECS but has no L1 counterpart (a specific sim's `charisma`), it is **not**
observable to the brain. Otherwise we train on a signal the training simulator cannot
provide and the policy silently learns to read noise.

---

## 8. Explicit non-goals

- **Sims are never ML.** They are scripted (utility / behaviour tree / GOAP),
  because their behaviour is not adversarial, and because architecture §9 shows sims
  are already the dominant per-tick cost — the last place to add a forward pass.
- **The district never enumerates its sims.** No O(N) command loops from any brain.
- **No policy multiplies an outcome** (I2).
- **No agent-based demography inside `ml_training/`** (§7).

---

## 9. Implementation order

Each step is independently verifiable, and each unblocks the next. Nothing before
step 1 is meaningful, because until sims can satisfy a need, happiness has nothing to
rest on.

Steps 1–2 are **game-side** (Rust) and remain open. Steps 3–5 are done **in the
training simulator**, which is what the RL work depends on; the game will need its
own per-sim implementation of the same four levers later.

| # | Step | Acceptance test |
|---|---|---|
| 1 | **The `Eat` loop.** Something must *increase* `diet_level`. Today nothing in `src/` calls `.increase()` on a need — the only `.increase()` in the whole codebase is `construction_progress`. Sims are spawned with `PersonStats::init()`/`PersonNeeds::init()` = 100 (`spawners/areas/population.rs`, and `spawners/buildings/newborn.rs` for births); `diet_level` decays 0.5/s, crosses the critical threshold of 10 at `t = 180 s`, then `life_points` falls 2/s to zero by `t ≈ 230 s`. That is before `kid_duration = 300 s`: **no sim in this game has ever reached adulthood, newborns included.** | a population survives past 300 s and produces a second generation |
| 2 | Utility selection over `{Eat, Rest, Socialize, Study, Work}`, sourced from buildings | needs oscillate around a setpoint instead of monotonically decaying |
| 3 | ✅ **(sim)** `happiness` is a real district variable, derived from conditions + policy costs | the hardcoded `happiness = 75.0` is gone; `BASE_HAPPINESS` is the baseline it recovers to |
| 4 | ✅ **(sim)** the four levers, with real effects: rationing, fertility modifier, childbirth mask, conscription | `test_no_policy_is_cost_only`; I1 holds (`test_rationing_is_not_a_ban_on_eating`) |
| 5 | ✅ **(sim)** conditional fertility | `test_conditions_dominate_the_incentive`; I2 holds |
| 6 | ✅ **(sim, partial)** age cohorts: kids don't work, elders work at 0.45. Attractiveness matching for jobs is still owed | `get_available_workers()` no longer counts children as labour |
| 7 | Calibration protocol (§7) | L1 tracks the ECS within tolerance; fidelity test in CI. **Blocked: the game does not exist yet** |
| 8 | Resource veins + district plot map + `BUILD(type, tile)` (§6) | spatial agent beats the flat 22-feature baseline |
| 9 | ✅ **(sim)** policies exposed to the district brain: cohorts, `happiness` and the active policies in the observation (22→29); the three district-scoped levers as actions (20→27) | BC v3 uses `SET_POPULATION_RATE_INCREASE` and beats the policy-blind BC v2 on both aggressive bots. See **D11** |

**Every lever must be sized against the episode.** Three constants were dead on
arrival because a "per hour" rate met a half-hour episode. See **D11** — it is the
first thing to check before adding an action.

Steps 1–3 are prerequisites for *everything* — including the answer to
"are policies enough?", which stays **no** until `happiness` is a real variable
and sims actually choose.

---

## 10. Open questions

1. **Does the district plot map belong to the district brain or to the macro?**
   Placing a *warehouse* founds a district (macro). Placing a farm inside one is
   district-tier. The boundary is clean in principle; the tile grid is shared.
2. **Elder productivity** (`elder_productivity` in `person.json`) exists but the
   training sim has no elders. L1 fixes this; is a 3-cohort split enough, or do
   skills demand a 4th dimension?
3. **Profession assignment.** `PersonProfession` is only ever *read*
   (`src/systems/factions/stats.rs`), never assigned. Who decides a sim's
   profession — the sim (by aptitude, an L2 target), or the district (an L4
   reassignment)? This determines whether `forced_conscription` is a special case or
   the general mechanism.
4. **Happiness aggregation.** `Area.happiness_level` is documented as the *median* of
   its people. Median is robust to a starving minority — which may be exactly the
   signal a district brain needs to see. Median or mean?
