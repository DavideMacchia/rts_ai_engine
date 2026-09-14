# RTS AI — Complete Architecture

> Status: living document. Last major revision: 2026-07-10
> (district-tier correction + region-graph map design).
> For *how the current training pipeline works* see
> [rl_training_system.md](rl_training_system.md); for *why specific choices were
> made* see [design_decisions.md](design_decisions.md); for *how a district steers
> self-managing sims, and how we simulate them cheaply enough to train on*, see
> [population_and_policies.md](population_and_policies.md).

---

## 1. The tiered agent, aligned to the game

An RTS/empire game asks the AI to reason at very different scales. Instead of one
flat brain, we use a **hierarchy that mirrors the game's own organization**
(empire → districts → units). Aligning the AI hierarchy with the game's real
structure means the abstraction boundaries are meaningful, not artificial.

```
                 ┌───────────────────────────────────────┐
                 │            MACRO BRAIN (RL)             │  ← NOT BUILT YET
                 │  region graph: discovered resources,    │
                 │  own/enemy districts, terrain, distances│
                 │  Decides WHERE to place warehouses      │
                 │  (= founds districts), allocates sims   │
                 │  and forces, coordinates, empire policy │
                 └───────┬─────────────────────┬───────────┘
             goals ▼                            ▼ goals
     ┌──────────────────────────┐   ┌──────────────────────────┐
     │  CIVILIAN DISTRICT        │   │   MILITARY CAMP           │  ← NOT BUILT YET
     │  (utility / rules)        │   │   (RL)                    │
     │  economy, happiness,      │   │   offensives, territory   │
     │  production; GARRISON      │   │   control; MOBILE FORCE   │
     │  (defensive); civil policy │   │   (offensive); mil policy │
     └───────────┬──────────────┘   └───────────┬──────────────┘
         commands ▼                              ▼ commands
     ┌───────────────────────────────────────────────────────────┐
     │              UNITS — individual AI (scripted)               │  ← NOT BUILT YET
     │   behavior trees / utility / GOAP; execute concrete orders  │
     │   exclusive ownership, transferable between district & camp │
     └───────────────────────────────────────────────────────────┘
```

**Three tiers, but only the adversarial ones learn.** The organizational hierarchy
(how many tiers) is separate from the learned surface (which tiers are ML). Keeping
the learned surface small is both simpler and cheaper at runtime.

---

## 2. What actually exists today — and it is NOT the macro brain

> **Stage 1 is done.** `Faction` is now a container of `District`s and every manager
> works per-district. The game still starts each faction with exactly one district, so
> everything below still describes the *behaviour* of the simulator; what changed is
> that the second district is now expressible.

> **Correction.** Earlier revisions of this document called the current agent "the
> macro brain". That was wrong.

The simulator models a faction as **one undifferentiated blob**: a single
settlement, one economy, one warehouse. In a **one-district world the three tiers
collapse into a single agent** — there is nothing to coordinate, and strategic
decisions are expressed directly as concrete build orders.

What we have trained is a **single-settlement strategic agent**:

| | What it does | Which tier |
|---|---|---|
| Observation | faction totals (resources, population, building counts) | *not* per-district summaries → **not macro** |
| Actions | `BUILD_FARM`, `BUILD_BARRACKS`, `TRAIN_SOLDIER`, `ATTACK` … | concrete → **district tier** |
| Objective | win the game | macro tier |

So: **its action space belongs to the district tier; only its objective is macro.**
The real macro brain **cannot exist until there are ≥2 districts to coordinate** —
its characteristic actions (found a district, allocate sims, transfer forces) are
all no-ops with a single district.

**Consequences (all good news):**
- Today's agent + the scripted bots (`NormalBot`, `AggressiveBot`) are **prototypes
  of the district tier**. `AggressiveBot` is literally a scripted settlement policy —
  which is exactly what we decided civilian districts should be (utility AI).
- **Today's observation is already a district observation.** Resources, buildings,
  population of a one-settlement faction *are* a district's local state. The refactor
  to districts was exactly that: move `Faction` state → `District` state, and make
  `Faction` a container of districts (done in Stage 1).
- The **infrastructure** (env, action masking, delta rewards, BC + critic warm-up,
  session logging, opponents) is tier-agnostic and transfers 100%.
- The **policy weights** do not transfer (they are tied to a 22-feature
  single-settlement observation). Retraining is cheap (~3 min per 500k steps).

---

## 3. The macro tier: the warehouse *is* the district

The core game rule that gives the macro a genuine decision:

> **Each district has at most one warehouse, and every warehouse founds a new
> district.** Placing a warehouse on the map = founding a district.

So the macro brain:
1. Chooses **where on the map** to place the first warehouse (founds district #1),
2. Assigns the first generated sims to it,
3. Later, with a full overview, decides **when to expand** and **where** to found the
   next civilian district or military camp.

**Defeat condition:** a faction loses when it loses **all** its warehouses. This
generalizes today's `is_defeated() == (warehouse count == 0)` for free once
warehouses are per-district — losing a district is a wound, not death, and expansion
becomes strategically central (redundancy + survival).

**Resources:** **per-district stockpiles, with logistics.** Each warehouse holds its
district's stock; transfers between districts cost time/capacity proportional to
graph distance. (Implementation is staged — see §7: stock first, transfers second.)

---

## 4. The map: an attributed region graph

The macro needs spatial information at *district* granularity: discovered resources,
own/enemy districts and outposts, terrain features (e.g. rivers) that shape where to
settle, and distances between districts (which drive logistics).

That is exactly an **attributed graph** — attributes on **nodes and on edges**:

```python
@dataclass
class Region:                  # NODE
    id: int
    terrain: Terrain           # plains / hill / forest / water …
    deposits: dict[str, float] # resource type -> richness
    discovered_by: set[int]    # fog of war, per faction
    owner: Optional[int]
    district_id: Optional[int] # at most ONE district (= one warehouse) here

@dataclass
class Link:                    # EDGE
    a: int; b: int
    distance: float
    traversal_cost: float      # river crossing, mountains -> high cost
```

- **Rivers live on edges** (high traversal cost, or no edge without a bridge).
- **Logistics cost** = shortest path over the graph between two of your districts.
  This makes "how far apart to place districts" a real, measurable trade-off.
- **Fog of war** = the per-node `discovered_by` flag; undiscovered regions are hidden
  or masked from the macro's observation.

No tiles, no pathfinding at the macro tier. The map is modelled at exactly the
granularity the macro reasons about.

### Node-scoped ("pointer") actions

The macro's action space becomes a **logit per region**, which scales automatically
with the number of regions:

```
FOUND_CIVILIAN_DISTRICT(region)
FOUND_MILITARY_CAMP(region)
ASSIGN_POPULATION(district, n)
ESTABLISH_ROUTE(district_a, district_b)   # once logistics exists
NO_OP
```

Our existing **action-masking** infrastructure applies directly: you cannot found on
an undiscovered, already-occupied, unreachable, or unaffordable region.

### The network: attention over the graph

A variable number of regions + relational structure + edge attributes calls for
**attention over entities with a distance/edge bias** (a graph-attention network).
This is where the retained `TransformerExtractor` becomes the *right* tool — not over
22 flat scalars (where we correctly replaced it with an MLP), but over a variable set
of region nodes.

### The macro is an easier RL problem than the district agent

The district agent takes ~720 decisions per game, nearly all irrelevant
(`DO_NOTHING`), with victory far in the future — a credit-assignment nightmare that
cost us seven distinct bugs (see design_decisions).

The macro will take perhaps **5–20 decisions per game** — where to found, how many
sims to assign, when to expand. Few decisions, each heavy and consequential. That is
a much more tractable learning problem.

---

## 5. Unit command ownership

Because a unit can serve either a civilian district (defense) or a military camp
(offense), the central risk is two brains commanding the same unit at once. The rule
that resolves it:

> **Every unit has exactly ONE owner at any instant. Ownership is explicit and
> transferable.**

```
Unit produced in a civilian district → owner = that district (GARRISON, defensive)
Macro decides an offensive           → TRANSFER units to a camp (MOBILE FORCE, offensive)
Offensive ends                       → stay with the camp, or return to garrison
```

**Ownership transfer is a macro decision** (force allocation) — exactly what the
global view is for. The military camp is the *mobile military command group*;
civilian garrisons are the *stationary defensive force*.

---

## 6. Communication between tiers

Two directions and **two speeds**.

**Vertical (feudal).** Down: goals/commands (macro → districts: allocations,
objectives, intent; district → units: concrete orders). Up: summaries and *requests*
(units → districts aggregate → macro sees per-district summaries).

**Horizontal (district ↔ district): a star, mediated by the macro.** Districts do
*not* form a peer-to-peer mesh for strategic decisions (N² channels, no global view,
multi-agent incoherence).

> **The macro IS the coordination mechanism.** Districts report up; the macro, with
> the global view, coordinates and redistributes via commands down.

A practical implementation is a **blackboard**: districts post requests/offers (needs
iron / has 5 free units); the macro (or simple market-style matching) pairs them.

**Two speeds.** The macro decides on a slow clock (~3–5 s) — too slow for a district
under attack. So add a **fast local reactive channel**:

```
SLOW / strategic:  district → request → MACRO → allocation/command → districts
FAST / reactive:   attacked district → broadcast SOS → adjacent districts/camps
                   react immediately with their local (utility) logic
```

### Policies — ownership follows scope

A game policy (siege mode, conscription, minimum food, …) is simply **an action in
the action space of the brain whose scope it affects**:

| Policy | Scope | Owner |
|---|---|---|
| Minimum food, happiness levers | district | **civilian district model** |
| Siege mode, conscription | camp / military | **military camp model** |
| Empire-wide laws, cross-district effects | global | **macro** |

Policies carry trade-offs (siege −20 % happiness). Balancing local readiness vs
happiness *is* a district/camp-level decision.

> **But a policy is a pair (effect, cost), and today only the cost exists** — in both
> codebases a policy merely subtracts happiness. Sims are autonomous, so a district
> steers them through **four distinct levers** (affordance, incentive, prohibition,
> coercion), not one. See [population_and_policies.md](population_and_policies.md).

---

## 7. Roadmap

> **Superseded from Stage 2 onwards by
> [roadmap_war_economy_macro.md](roadmap_war_economy_macro.md).** The table below was
> written before the map existed. What it got wrong is the ORDER: it put the map and the
> macro before the war economy, and D28 measured the consequence — the map is inert while
> iron cannot become an army (`iron_ingots` are produced and consumed by nothing, and
> forcing every building onto its worst tile changes conquests by zero). The war economy —
> primary resources refined into weapons and armour that an army actually requires — comes
> first, then the camp tier, then a bigger map, then the macro brain.

The principle that keeps this tractable: **train one tier at a time, freezing the
others.** In Stage 3 the macro learns while the district brains stand still — few
steps, few decisions, a clean signal.

| Stage | Content | Control / test |
|---|---|---|
| 0 | Single-settlement agent (env, masking, delta rewards, BC + critic warm-up) | ✅ done — see §2 |
| 1 | **`Faction` = set of `District`s.** Start with exactly ONE district | ✅ done — BC policy still wins 10/10 vs all three bots; `tests/unit/test_districts.py` covers the multi-district paths |
| **2** | Region graph + map generation; the district lives on a region. Still one district | No behaviour change; the map exists but is inert. `District.location` is already the seam |
| 3 | **Macro v1**: `FOUND_DISTRICT(region)` + `ASSIGN_POPULATION`. District brains **frozen** (scripted/BC). Per-district stock, **no** transfers yet | First real macro learning, clean credit assignment |
| 4 | **Logistics**: routes and transfers, cost from the graph | The distance trade-off becomes real |
| 5 | **Military camps** as a district subtype; unit ownership/transfer; force allocation | |
| 6 | Fog of war / scouting (`discovered_by` exists from Stage 2) | |
| 7 | Self-play / co-training of the tiers | |

**Stage 1, in hindsight.** It touched every manager except `reward_calculator` (which
reads only faction-level aggregates, and rewards are faction-scoped anyway). The
perfect regression oracle held throughout: the BC policy still wins 10/10. The one
subtle decision — how `Faction` can expose aggregates without silently swallowing the
faction-level writes that pervade the codebase — is written up as **D10**.

> Note: Stage 3 delivers much of the intent with a fraction of the work — per-district
> stock alone already forces the macro to found districts *near resources*. Logistics
> (Stage 4) then adds the "how far apart" trade-off.

---

## 8. Interfaces & sim-to-real

1. **Macro ↔ District** — goals down, summaries/requests up (§6).
2. **District ↔ Units** — concrete orders down, state up; unit ownership token (§5).
3. **Combat oracle** — `CombatManager.execute_attack` is a **swappable placeholder**
   standing in for real combat. The macro must depend only on "army X vs army Y →
   outcome", never on the internals.
4. **Simulator ↔ Rust game** — the training sim is a fast approximation; the game is
   ground truth. Sharing `game_data/*.json` keeps economy/costs consistent.

---

## 9. Runtime performance

The ML tiers are **not** the bottleneck; per-unit simulation is — and that cost is
inherent to any RTS, ML or not.

- **Training is offline.** At runtime we only run **inference** (a forward pass).
  A small MLP is ~0.1–0.5 ms on CPU, less when batched.
- **Temporal abstraction** is the main lever: macro decides every ~3–5 s, districts
  ~1 s, only units tick often (and they're scripted).
- **Batch across factions and districts:** all decisions share the architecture, so
  run them as ONE batched forward pass → N factions ≈ free.
- **What would hurt (and mitigations):** running ML every frame (don't); many separate
  small NNs called individually (batch them, or use utility AI for civilian
  districts); one-frame spikes (stagger / time-slice AI on a budgeted thread); GPU for
  tiny nets (often slower than batched CPU).

Rough budget (30 Hz sim, ~33 ms/tick, ~8 factions × ~50 districts): macro <0.1 ms
amortized, districts ~0.5–1 ms amortized, units = the real cost.

**Caveat — multiplayer determinism.** If the game uses lockstep multiplayer, AI must
produce identical results on every machine; float NN inference can diverge. Options:
AI server-side only, or deterministic (fixed-point/quantized) inference.

---

## 10. Summary of decisions

- **Tiered hierarchy aligned to the game** (empire → districts → units). *Made.*
- **Only the adversarial tiers learn**: macro RL + military-camp RL; civilian
  districts utility; units scripted. *Made.*
- **What exists today is the DISTRICT tier**, not the macro. The macro is unbuildable
  until ≥2 districts exist. *Made (correction).*
- **The warehouse is the district seed**; the macro decides where to place it; a
  faction dies only when it loses **all** warehouses. *Made.*
- **Map = attributed region graph** (node + edge attributes; rivers on edges; fog of
  war per node). Macro actions are **node-scoped pointer actions** with masking.
  Network = attention over regions. *Made.*
- **Per-district stockpiles with logistics**, implemented in two steps (stock first,
  transfers second). *Made.*
- **Exclusive, transferable unit ownership**; transfers are macro decisions. *Made.*
- **Communication:** vertical feudal; horizontal **star, macro-mediated** (blackboard,
  no peer mesh); plus a fast local reactive SOS channel. *Made.*
- **Combat = swappable oracle** the macro must not peek inside. *Made.*
- **Train one tier at a time, freezing the others.** *Made.*
