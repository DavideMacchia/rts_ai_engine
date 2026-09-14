# Roadmap — the war economy, the camp tier, the bigger map, the macro brain

> Supersedes the coarse stage table in [architecture.md §7](architecture.md) for everything
> after the district tier. Written after D27/D28, and shaped by what those measurements
> actually said rather than by what we hoped they would say.

## Where this comes from: the measurement that forced it

D28 built the spatial action space — the agent chooses the tile it builds on — and it works
(tile-match 97%, placement 0.95 against the auto-placer's 0.96). Then the ablation asked
whether WHERE is worth anything at all: force every building onto its *worst* legal tile
instead of its best, cutting raw yield from 0.96 to 0.77.

**Conquests did not move** (19/17/17 vs 18/18/16). A 20% production cut changes nothing.

Two causes, and together they are this roadmap:

1. **The map has nothing worth owning.** The mine produces `iron_ore`, smelting turns it into
   `iron_ingots`, and *nothing in the game consumes them*. Soldiers cost wood and grain. The
   one genuinely scarce thing on the ground cannot be turned into anything that wins a game.
2. **Production is not the binding constraint.** Armies are capped by adult **men** (D22).
   Placement scales a resource that is not the one running out.

So the order below is not arbitrary. **The war economy comes first**, because until iron can
become an army, the map, the camps, and the macro brain are all decorating a decision that
does not exist.

---

## Stage A — The war economy: primary resources → derivatives → an actual army

**The idea.** An army is not conjured from wood and grain. It is *equipped*. Primary resources
are refined into **derivatives** — wooden weapons, iron weapons, armour — and a soldier cannot
exist without his equipment. This is what finally makes geography strategic: a plot with iron
can field a better army than a plot without one, and that is a reason to want ground.

**Chains** (they extend the existing `RESOURCE_PROCESSING`, which already has the shape):

```
forest    -> lumberyard -> wood ---> carpentry  -> wooden weapons  (cheap, weak, always available)
mountain  -> quarry     -> stone
iron vein -> mine       -> iron_ore -> smelting  -> iron_ingots -> blacksmith -> iron weapons
                                                                              -> armour
gold vein -> mine       -> gold  ---> (to be decided: pay/upkeep? luxury happiness?)
```

**The rule that gives it teeth:** `TRAIN_SOLDIER` consumes *equipment*, not just food and wood.
A man with a wooden spear and a man in iron mail are not the same soldier, and the difference
must show up where it matters — in `UNIT_STRENGTH`, i.e. in who wins the battle.

**Design constraints, learned the hard way:**
- **Anchor the ratios, never the numbers** (D13). How much stronger is an iron soldier than a
  wooden one? Express it as a ratio against the base soldier, and derive the rest.
- **Do not break neutrality silently** (D27). Introducing equipment must not, on day one,
  re-balance the game by accident. Give the base soldier a default (wooden) equipment so the
  existing balance is reproduced, then let iron be the *upside*.
- **Iron must be gated by the map, not by the wallet.** If you can simply buy your way to iron
  weapons, the ground is irrelevant again and we are back where D28 left us.

**Definition of done — and it is a real risk, not a formality:**
> Re-run the D28 ablation. Best placement vs worst placement must now *separate* on conquests.
> If it still does not, equipment has not been made to matter and the whole chain is another
> neutral lever — **do not build Stage B on top of an unmeasured Stage A.**

Second check: a plot **with** iron must beat a plot **without** it, holding everything else
fixed. That is the number that says the map became strategic.

---

## Stage B — Split the military into its own model: the CAMP district

**This was always the plan** (`agents/camp/`, architecture.md §5) and D25/D26 made the reason
concrete: the district brain is now doing two jobs badly at once — running a settlement and
running a war — and the war logic has been bolted onto a civil brain.

**The split, in the game's own terms** (`documentation/gameplay/area_system.md`: an Area is
centred on a **warehouse** *or* a **military camp**):

| | **Civil district** (exists) | **Camp district** (new tier) |
|---|---|---|
| Centred on | warehouse | military camp |
| Job | economy, population, equipment | war |
| Decisions | build, refine, raise and recruit men, **found camps**, send men + equipment to camps | move armies, engage, siege, retreat |
| Costs it pays | civilians spent to found a camp; men spent to fill it | |

**The civil district's job becomes explicit:**
1. extract primary resources and refine them into derivatives (Stage A);
2. raise people, recruit them into soldiers at the barracks;
3. **found camp districts** — spending civilians to do it;
4. ship soldiers *and their equipment* to the camps.

**War then moves from the camps, not from the city.** This is the natural home for everything
D25/D26 uncovered — the march, the army that leaves home and cannot defend it, the commit
decision — and it is where "how much do I send" (the open lever from D26: `ATTACK` currently
commits the *whole* garrison, because the action space has no "attack with N") finally belongs.

**Interface (keep it narrow, or the tiers stop being separable):** goals down, summaries up
(architecture.md §6). The camp asks for men and equipment; the city decides whether to pay.
The combat oracle stays swappable — nothing above it may depend on its internals.

**Training discipline:** train one tier at a time, freezing the others. Clone the camp tier
from a scripted expert first (BC has worked every time here; pure RL has never discovered a
long action sequence on its own — D4).

---

## Stage C — A bigger map

Inevitable once Stage B exists, and only worth paying for once it does. The 64×64 training map
holds one district per side; it cannot hold **resources that are near and far**, several
districts, allies and enemies.

**What must NOT be lost when it grows:**
- **Symmetry** (D27). The map is rotationally symmetric *by construction*, and the measured
  consequence is a 0.0000 difference in starting plot quality. On a random map the first
  settler took the better ground (water in 84% of faction 0's plots vs 58% of faction 1's). A
  permanent invisible head-start would poison every measurement made from then on. A bigger map
  with more districts needs symmetry that survives *n* settlements — plan the generator for it.
- **The ratios** (D13/D27). `DISTRICT_RADIUS : SETTLEMENT_SEPARATION : MAP_SIZE`. Growing the
  map means deciding how far "far" is, in marches — and the march is derived from distance, so
  the tempo of the whole war moves with it. Change the ratio deliberately, and measure it.
- **Cost.** World generation is ~39 ms; it runs every episode. A much bigger map, generated
  per reset, will show up in training throughput. Budget it before it surprises us.

**New affordances the size buys:** near vs far resources (a real logistics trade-off — the
"how far apart" question architecture.md §7 Stage 4 wanted), several allied/enemy districts,
and the ground for fog of war later.

---

## Stage D — The macro brain

**Only now**, because it needs something to decide *between*: multiple districts, camps to
found, resources near and far. The macro is the AI that governs the districts — `FOUND_DISTRICT
(region)`, `FOUND_CAMP`, allocation of population and equipment between them, logistics.

**What is already settled about it:**
- **Pointer actions, factored and masked** — `FOUND_DISTRICT(region)`, not a flattened
  `Discrete(types × places)`. D28 confirmed this the hard way for `BUILD(type, tile)`:
  factoring is what lets anything learned about a place transfer to the next thing put on it.
- **Attention over the region graph is the right tool here** — a variable-size *set* of
  regions is exactly what the retained `TransformerExtractor` is for, and exactly what the
  district's fixed 13×13 lattice was *not* (there, a convnet won; D28).
- **The macro is an easier RL problem than the district** (architecture.md §4): few decisions,
  each one consequential.
- Train it with the district and camp brains **frozen**, for clean credit assignment.

---

## The order, and why it is this order

```
A. war economy  ──▶  B. camp tier  ──▶  C. bigger map  ──▶  D. macro brain
   (gives the ground a prize)   (gives war its own brain)
   (gives equipment a purpose)              (gives the macro something to choose between)
```

A is first because it is the only one that can be *measured* today, and because B, C and D all
assume it: a camp is worth founding only if it can be equipped, a bigger map is worth walking
only if the far ground is worth owning, and a macro brain is worth training only if its choices
differ.

## The discipline this roadmap is written under

These are not slogans; each one is a bill we have already paid.

1. **Ablate every new lever against CONQUESTS, never against a proxy.** Roughly half of them
   have turned out neutral or harmful: the policy levers (D15), the defender-reinforcement
   model (D25), and the spatial action space itself (D28). A clone once reached 99%
   action-match by *never attacking*.
2. **n=20 is noise.** The same configuration has sampled 3/20 and 13/20. Use n≥40, paired
   seeds, and hold the opponent fixed when ablating the agent.
3. **A mirror match is a coin flip.** An expert measured against its own bot is capped near
   50%. That is not a skill level.
4. **A new subsystem must be neutral on introduction**, or nothing measured before it can be
   compared with anything measured after (D27).
5. **Check whether the mechanic is even wired up.** The biggest win of the whole map effort was
   a *dead constant* (`WAREHOUSE_DESTRUCTION_THRESHOLD`, read by nothing — D24), and the
   biggest disappointment was a dead resource (`iron_ingots`, consumed by nothing — D28).
